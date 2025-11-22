"""
Utilidades generales: configuración, logging, checkpoints, semillas
"""
import random
import torch
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging


def setup_logging(log_file: str = None):
    """Configura logging para el proyecto"""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Establece semillas para reproducibilidad completa.
    
    Args:
        seed: Semilla a usar (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Para reproducibilidad completa (puede ser más lento)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Para MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    config: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = None
):
    """
    Guarda checkpoint del modelo.
    
    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Época actual
        step: Step actual
        loss: Pérdida actual
        config: Configuración del modelo
        checkpoint_dir: Directorio donde guardar
        filename: Nombre del archivo (opcional)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint guardado en: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu'
):
    """
    Carga checkpoint del modelo.
    
    Args:
        checkpoint_path: Ruta al checkpoint
        model: Modelo donde cargar los pesos
        optimizer: Optimizador (opcional)
        device: Dispositivo donde cargar
        
    Returns:
        Diccionario con información del checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Checkpoint cargado desde: {checkpoint_path}")
    logging.info(f"Época: {checkpoint['epoch']}, Step: {checkpoint['step']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


def get_device(device: str = 'auto') -> torch.device:
    """
    Obtiene el dispositivo a usar (CPU, CUDA, MPS).
    
    Args:
        device: 'auto', 'cpu', 'cuda', 'mps'
        
    Returns:
        torch.device
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    device = torch.device(device)
    logging.info(f"Usando dispositivo: {device}")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Cuenta el número de parámetros entrenables del modelo.
    
    Args:
        model: Modelo de PyTorch
        
    Returns:
        Número de parámetros entrenables
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """
    Formatea un número grande (ej: 1234567 -> 1.23M)
    
    Args:
        num: Número a formatear
        
    Returns:
        String formateado
    """
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


class AverageMeter:
    """Computa y almacena el promedio y valor actual"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_directory_structure():
    """Crea la estructura de directorios necesaria"""
    dirs = [
        'checkpoints',
        'outputs',
        'data/raw',
        'data/processed',
        'logs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        # Crear .gitkeep para mantener directorios vacíos en git
        gitkeep_path = os.path.join(dir_path, '.gitkeep')
        Path(gitkeep_path).touch()
