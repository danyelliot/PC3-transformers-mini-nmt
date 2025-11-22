"""
Evaluación del modelo con métricas estándar de traducción.

Métricas implementadas:
- sacreBLEU
- chrF++
- Perplejidad
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
from typing import List, Dict
import math
from tqdm import tqdm

from .models.transformer import Transformer, TransformerConfig
from .data import load_data, create_dataloaders, Vocabulary
from .decoding import Decoder, DecodingConfig
from .utils import set_seed, get_device, load_checkpoint, load_config, setup_logging

logger = logging.getLogger(__name__)

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logger.warning("sacrebleu no instalado. Algunas métricas no estarán disponibles.")


class Evaluator:
    """
    Clase para evaluar el modelo Transformer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        device: torch.device,
        decoding_config: DecodingConfig = None
    ):
        """
        Args:
            model: Modelo Transformer
            test_loader: DataLoader de test
            src_vocab: Vocabulario fuente
            tgt_vocab: Vocabulario objetivo
            device: Dispositivo
            decoding_config: Configuración de decodificación
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        if decoding_config is None:
            decoding_config = DecodingConfig()
        
        self.decoder = Decoder(
            model,
            decoding_config,
            bos_token_id=tgt_vocab.sos_idx,
            eos_token_id=tgt_vocab.eos_idx,
            pad_token_id=tgt_vocab.pad_idx,
            device=device
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    
    @torch.no_grad()
    def calculate_perplexity(self) -> float:
        """
        Calcula la perplejidad en el conjunto de test.
        
        Returns:
            Perplejidad
        """
        total_loss = 0.0
        total_tokens = 0
        
        for src, tgt in tqdm(self.test_loader, desc="Calculando perplejidad"):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward
            logits = self.model(src, tgt_input)
            
            # Loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # Contar tokens válidos (no padding)
            n_tokens = (tgt_output != self.tgt_vocab.pad_idx).sum().item()
            
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def translate_batch(self, src: torch.Tensor) -> List[List[int]]:
        """
        Traduce un batch de secuencias.
        
        Args:
            src: (batch, src_len)
            
        Returns:
            Lista de traducciones (índices)
        """
        src = src.to(self.device)
        translations = self.decoder.decode(src)
        return translations
    
    def indices_to_text(self, indices: List[int], vocab: Vocabulary) -> str:
        """
        Convierte lista de índices a texto.
        
        Args:
            indices: Lista de índices
            vocab: Vocabulario
            
        Returns:
            Texto como string
        """
        tokens = []
        for idx in indices:
            if idx == vocab.eos_idx:
                break
            if idx not in [vocab.pad_idx, vocab.sos_idx]:
                tokens.append(vocab.idx2token.get(idx, vocab.unk_token))
        
        return ' '.join(tokens)
    
    def calculate_bleu(self, max_samples: int = None) -> Dict[str, float]:
        """
        Calcula BLEU score.
        
        Args:
            max_samples: Número máximo de muestras a evaluar (None = todas)
            
        Returns:
            Diccionario con scores BLEU
        """
        if not SACREBLEU_AVAILABLE:
            logger.warning("sacrebleu no disponible, retornando BLEU = 0")
            return {'bleu': 0.0}
        
        hypotheses = []
        references = []
        
        n_samples = 0
        for src, tgt in tqdm(self.test_loader, desc="Generando traducciones"):
            if max_samples and n_samples >= max_samples:
                break
            
            # Traducir
            translations = self.translate_batch(src)
            
            # Convertir a texto
            for i in range(len(translations)):
                # Hipótesis (traducción del modelo)
                hyp = self.indices_to_text(translations[i], self.tgt_vocab)
                hypotheses.append(hyp)
                
                # Referencia (traducción real)
                ref = self.indices_to_text(tgt[i].tolist(), self.tgt_vocab)
                references.append([ref])  # sacreBLEU espera lista de referencias
                
                n_samples += 1
                if max_samples and n_samples >= max_samples:
                    break
        
        # Calcular BLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, references)
        
        results = {
            'bleu': bleu.score,
            'bleu_1': bleu.precisions[0],
            'bleu_2': bleu.precisions[1],
            'bleu_3': bleu.precisions[2],
            'bleu_4': bleu.precisions[3],
        }
        
        return results
    
    def calculate_chrf(self, max_samples: int = None) -> float:
        """
        Calcula chrF++ score.
        
        Args:
            max_samples: Número máximo de muestras
            
        Returns:
            chrF++ score
        """
        if not SACREBLEU_AVAILABLE:
            logger.warning("sacrebleu no disponible, retornando chrF = 0")
            return 0.0
        
        hypotheses = []
        references = []
        
        n_samples = 0
        for src, tgt in tqdm(self.test_loader, desc="Calculando chrF++"):
            if max_samples and n_samples >= max_samples:
                break
            
            translations = self.translate_batch(src)
            
            for i in range(len(translations)):
                hyp = self.indices_to_text(translations[i], self.tgt_vocab)
                hypotheses.append(hyp)
                
                ref = self.indices_to_text(tgt[i].tolist(), self.tgt_vocab)
                references.append([ref])
                
                n_samples += 1
                if max_samples and n_samples >= max_samples:
                    break
        
        chrf = sacrebleu.corpus_chrf(hypotheses, references)
        
        return chrf.score
    
    def generate_samples(self, n_samples: int = 10) -> List[Dict[str, str]]:
        """
        Genera muestras de traducción para inspección manual.
        
        Args:
            n_samples: Número de muestras
            
        Returns:
            Lista de diccionarios con source, reference, hypothesis
        """
        samples = []
        
        for src, tgt in self.test_loader:
            if len(samples) >= n_samples:
                break
            
            translations = self.translate_batch(src)
            
            for i in range(min(len(translations), n_samples - len(samples))):
                source_text = self.indices_to_text(src[i].tolist(), self.src_vocab)
                reference_text = self.indices_to_text(tgt[i].tolist(), self.tgt_vocab)
                hypothesis_text = self.indices_to_text(translations[i], self.tgt_vocab)
                
                samples.append({
                    'source': source_text,
                    'reference': reference_text,
                    'hypothesis': hypothesis_text
                })
        
        return samples
    
    def evaluate_all(self, max_samples: int = None) -> Dict[str, float]:
        """
        Ejecuta todas las evaluaciones.
        
        Args:
            max_samples: Número máximo de muestras para BLEU/chrF
            
        Returns:
            Diccionario con todas las métricas
        """
        logger.info("Iniciando evaluación completa...")
        
        results = {}
        
        # Perplejidad
        logger.info("Calculando perplejidad...")
        ppl = self.calculate_perplexity()
        results['perplexity'] = ppl
        logger.info(f"Perplejidad: {ppl:.2f}")
        
        # BLEU
        if SACREBLEU_AVAILABLE:
            logger.info("Calculando BLEU...")
            bleu_results = self.calculate_bleu(max_samples)
            results.update(bleu_results)
            logger.info(f"BLEU: {bleu_results['bleu']:.2f}")
            
            # chrF++
            logger.info("Calculando chrF++...")
            chrf = self.calculate_chrf(max_samples)
            results['chrf'] = chrf
            logger.info(f"chrF++: {chrf:.2f}")
        
        # Ejemplos
        logger.info("\nEjemplos de traducción:")
        samples = self.generate_samples(5)
        for i, sample in enumerate(samples, 1):
            logger.info(f"\nEjemplo {i}:")
            logger.info(f"  Source:    {sample['source']}")
            logger.info(f"  Reference: {sample['reference']}")
            logger.info(f"  Hypothesis: {sample['hypothesis']}")
        
        return results


def main():
    """Función principal de evaluación."""
    parser = argparse.ArgumentParser(description="Evaluar Transformer NMT")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Configuración")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Checkpoint del modelo")
    parser.add_argument("--split", type=str, default="test", choices=['val', 'test'], help="Split a evaluar")
    parser.add_argument("--max_samples", type=int, default=None, help="Máximo de muestras para BLEU/chrF")
    parser.add_argument("--strategy", type=str, default="beam", choices=['greedy', 'beam', 'topk', 'topp'], help="Estrategia de decodificación")
    parser.add_argument("--beam_size", type=int, default=4, help="Tamaño del beam")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging("logs/eval.log")
    config = load_config(args.config)
    set_seed(config.get('system', {}).get('seed', 42))
    device = get_device(config.get('system', {}).get('device', 'auto'))
    
    # Cargar vocabularios
    import pickle
    logger.info("Cargando vocabularios...")
    
    with open("data/processed/vocab_src.pkl", 'rb') as f:
        vocab_data = pickle.load(f)
        if isinstance(vocab_data, Vocabulary):
            src_vocab = vocab_data
        else:
            src_vocab = Vocabulary()
            src_vocab.token2idx = vocab_data['token2idx']
            src_vocab.idx2token = vocab_data['idx2token']
    
    with open("data/processed/vocab_tgt.pkl", 'rb') as f:
        vocab_data = pickle.load(f)
        if isinstance(vocab_data, Vocabulary):
            tgt_vocab = vocab_data
        else:
            tgt_vocab = Vocabulary()
            tgt_vocab.token2idx = vocab_data['token2idx']
            tgt_vocab.idx2token = vocab_data['idx2token']
    
    # Cargar datos
    logger.info("Cargando datos...")
    splits = load_data(
        data_dir="data/raw",
        max_samples=config.get('data', {}).get('max_samples', 20000),
        train_split=config.get('data', {}).get('train_split', 0.8),
        val_split=config.get('data', {}).get('val_split', 0.1),
        test_split=config.get('data', {}).get('test_split', 0.1)
    )
    
    dataloaders = create_dataloaders(
        splits,
        src_vocab,
        tgt_vocab,
        batch_size=config.get('data', {}).get('batch_size', 32),
        max_length=config.get('model', {}).get('max_seq_length', 128),
        num_workers=0  # Usar 0 para evaluación
    )
    
    # Actualizar config
    config['model']['vocab_size_src'] = len(src_vocab)
    config['model']['vocab_size_tgt'] = len(tgt_vocab)
    config['model']['pad_idx'] = src_vocab.pad_idx
    
    # Crear modelo
    logger.info("Cargando modelo...")
    model_config = TransformerConfig(**config['model'])
    model = Transformer(model_config)
    
    # Cargar checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    
    # Configuración de decodificación
    decoding_config = DecodingConfig(
        strategy=args.strategy,
        beam_size=args.beam_size,
        length_penalty=config.get('decoding', {}).get('length_penalty', 0.6),
        repetition_penalty=config.get('decoding', {}).get('repetition_penalty', 1.2),
        temperature=config.get('decoding', {}).get('temperature', 1.0),
        top_k=config.get('decoding', {}).get('top_k', 50),
        top_p=config.get('decoding', {}).get('top_p', 0.92),
    )
    
    # Crear evaluador
    evaluator = Evaluator(
        model,
        dataloaders[args.split],
        src_vocab,
        tgt_vocab,
        device,
        decoding_config
    )
    
    # Evaluar
    results = evaluator.evaluate_all(max_samples=args.max_samples)
    
    # Guardar resultados
    import json
    output_path = f"outputs/eval_results_{args.split}_{args.strategy}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResultados guardados en {output_path}")


if __name__ == "__main__":
    import os
    main()
