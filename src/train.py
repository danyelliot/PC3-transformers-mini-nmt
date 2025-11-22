"""
Loop de entrenamiento para el modelo Transformer NMT.

Incluye:
- Warmup learning rate schedule
- Cosine decay
- Gradient clipping
- Mixed precision training (AMP)
- Early stopping
- Checkpointing
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import time
import math
import yaml
import argparse
import logging
from tqdm import tqdm

from .models.transformer import Transformer, TransformerConfig
from .data import load_data, create_dataloaders, Vocabulary
from .utils import (
    set_seed, get_device, count_parameters, format_number,
    AverageMeter, save_checkpoint, load_config, setup_logging
)

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """
    Learning rate scheduler con warmup lineal y cosine decay.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        """
        Args:
            optimizer: Optimizador de PyTorch
            warmup_steps: Pasos de warmup
            total_steps: Total de pasos de entrenamiento
            min_lr: Learning rate mínimo
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Actualiza el learning rate."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Warmup lineal
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        """Retorna el learning rate actual."""
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """
    Clase para manejar el entrenamiento del Transformer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: torch.device,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary
    ):
        """
        Args:
            model: Modelo Transformer
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            config: Configuración de entrenamiento
            device: Dispositivo (CPU/CUDA)
            src_vocab: Vocabulario fuente
            tgt_vocab: Vocabulario objetivo
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Configuración de entrenamiento
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 30)
        self.grad_clip = train_config.get('max_grad_norm', 1.0)
        self.label_smoothing = train_config.get('label_smoothing', 0.1)
        self.use_amp = train_config.get('use_amp', True)
        self.early_stopping_patience = train_config.get('early_stopping_patience', 5)
        self.save_every = train_config.get('save_every', 5)
        
        # Sistema
        system_config = config.get('system', {})
        self.checkpoint_dir = system_config.get('checkpoint_dir', 'checkpoints')
        self.log_every = system_config.get('log_every', 100)
        self.eval_every = system_config.get('eval_every', 500)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Loss function con label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tgt_vocab.pad_idx,
            label_smoothing=self.label_smoothing
        )
        
        # Optimizador
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=train_config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = train_config.get('warmup_steps', 1000)
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer inicializado")
        logger.info(f"Parámetros del modelo: {format_number(count_parameters(model))}")
        logger.info(f"Optimizador: Adam, LR: {train_config.get('learning_rate', 1e-4)}")
        logger.info(f"Scheduler: Warmup ({warmup_steps} steps) + Cosine Decay")
        logger.info(f"Mixed Precision: {self.use_amp}")
    
    def train_epoch(self, epoch: int):
        """
        Entrena una época.
        
        Args:
            epoch: Número de época actual
            
        Returns:
            Loss promedio de la época
        """
        self.model.train()
        epoch_loss = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Separar input y target
            # Input: <SOS> token1 token2 ... tokenN
            # Target: token1 token2 ... tokenN <EOS>
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass con mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(src, tgt_input)
                    
                    # Calcular loss
                    # logits: (batch, seq_len, vocab_size)
                    # tgt_output: (batch, seq_len)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_output.reshape(-1)
                    )
                
                # Backward con scaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                # Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Update scheduler
            lr = self.scheduler.step()
            
            # Update metrics
            epoch_loss.update(loss.item(), src.size(0))
            self.global_step += 1
            
            # Logging
            if self.global_step % self.log_every == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{lr:.6f}'
                })
            
            # Validación intermedia
            if self.global_step % self.eval_every == 0:
                val_loss = self.validate()
                logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}")
                self.model.train()
        
        return epoch_loss.avg
    
    @torch.no_grad()
    def validate(self):
        """
        Valida el modelo.
        
        Returns:
            Loss promedio en validación
        """
        self.model.eval()
        val_loss = AverageMeter()
        
        for src, tgt in self.val_loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            if self.use_amp:
                with autocast():
                    logits = self.model(src, tgt_input)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_output.reshape(-1)
                    )
            else:
                logits = self.model(src, tgt_input)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
            
            val_loss.update(loss.item(), src.size(0))
        
        return val_loss.avg
    
    def train(self):
        """
        Loop principal de entrenamiento.
        """
        logger.info("Iniciando entrenamiento...")
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # Entrenar época
            train_loss = self.train_epoch(epoch)
            
            # Validar
            val_loss = self.validate()
            
            # Calcular perplejidad
            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            logger.info(f"\nEpoch {epoch}/{self.epochs}")
            logger.info(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            logger.info(f"Time: {epoch_time:.2f}s")
            
            # Guardar checkpoint
            if epoch % self.save_every == 0 or epoch == self.epochs:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.global_step,
                    val_loss,
                    self.config,
                    self.checkpoint_dir,
                    f"checkpoint_epoch{epoch}.pt"
                )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Guardar mejor modelo
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.global_step,
                    val_loss,
                    self.config,
                    self.checkpoint_dir,
                    "best_model.pt"
                )
                logger.info(f"Mejor modelo guardado (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                logger.info(f"Early stopping patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping activado")
                    break
        
        logger.info("Entrenamiento completado")


def main():
    """Función principal de entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrenar Transformer NMT")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Ruta al archivo de configuración")
    parser.add_argument("--resume", type=str, default=None, help="Ruta a checkpoint para resumir entrenamiento")
    parser.add_argument("--profile", action="store_true", help="Activar profiling")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging("logs/train.log")
    config = load_config(args.config)
    
    # Semilla para reproducibilidad
    set_seed(config.get('system', {}).get('seed', 42))
    
    # Device
    device = get_device(config.get('system', {}).get('device', 'auto'))
    
    # Cargar datos
    logger.info("Cargando datos...")
    
    # Cargar vocabularios
    import pickle
    with open("data/processed/vocab_src.pkl", 'rb') as f:
        src_vocab = pickle.load(f)
        if not isinstance(src_vocab, Vocabulary):
            # Si es un dict, reconstruir
            vocab_obj = Vocabulary()
            vocab_obj.token2idx = src_vocab['token2idx']
            vocab_obj.idx2token = src_vocab['idx2token']
            src_vocab = vocab_obj
    
    with open("data/processed/vocab_tgt.pkl", 'rb') as f:
        tgt_vocab = pickle.load(f)
        if not isinstance(tgt_vocab, Vocabulary):
            vocab_obj = Vocabulary()
            vocab_obj.token2idx = tgt_vocab['token2idx']
            vocab_obj.idx2token = tgt_vocab['idx2token']
            tgt_vocab = vocab_obj
    
    # Cargar splits
    splits = load_data(
        data_dir="data/raw",
        max_samples=config.get('data', {}).get('max_samples', 20000),
        train_split=config.get('data', {}).get('train_split', 0.8),
        val_split=config.get('data', {}).get('val_split', 0.1),
        test_split=config.get('data', {}).get('test_split', 0.1)
    )
    
    # Crear DataLoaders
    dataloaders = create_dataloaders(
        splits,
        src_vocab,
        tgt_vocab,
        batch_size=config.get('data', {}).get('batch_size', 32),
        max_length=config.get('model', {}).get('max_seq_length', 128),
        num_workers=config.get('data', {}).get('num_workers', 2)
    )
    
    # Actualizar config con tamaños de vocabulario reales
    config['model']['vocab_size_src'] = len(src_vocab)
    config['model']['vocab_size_tgt'] = len(tgt_vocab)
    config['model']['pad_idx'] = src_vocab.pad_idx
    
    # Crear modelo
    logger.info("Creando modelo...")
    model_config = TransformerConfig(**config['model'])
    model = Transformer(model_config)
    
    # Resumir entrenamiento si se especifica
    if args.resume:
        from .utils import load_checkpoint
        load_checkpoint(args.resume, model, device=device)
    
    # Crear trainer
    trainer = Trainer(
        model,
        dataloaders['train'],
        dataloaders['val'],
        config,
        device,
        src_vocab,
        tgt_vocab
    )
    
    # Profiling
    if args.profile:
        logger.info("Modo profiling activado")
        from torch.profiler import profile, ProfilerActivity
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # Entrenar solo 10 batches para profiling
            for i, (src, tgt) in enumerate(dataloaders['train']):
                if i >= 10:
                    break
                
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits = model(src, tgt_input)
                loss = trainer.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
                loss.backward()
        
        # Guardar trace
        prof.export_chrome_trace("trace.json")
        logger.info("Trace guardado en trace.json (abrir en chrome://tracing)")
        
        # Imprimir tabla
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    else:
        # Entrenamiento normal
        trainer.train()


if __name__ == "__main__":
    main()
