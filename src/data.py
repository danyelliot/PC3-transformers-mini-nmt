"""
Carga y procesamiento de datos para NMT.

Descarga el dataset Tatoeba español-inglés, tokeniza y crea DataLoaders.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional, Dict
import urllib.request
import zipfile
import gzip
from collections import Counter
import pickle
import logging

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Vocabulario para tokenización.
    
    Mantiene mapeo entre tokens y índices.
    """
    
    def __init__(self, freq_threshold: int = 2):
        """
        Args:
            freq_threshold: Frecuencia mínima para incluir un token
        """
        self.freq_threshold = freq_threshold
        
        # Tokens especiales
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        # Mapeos
        self.token2idx = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.token_freqs = Counter()
    
    def build_vocabulary(self, sentences: List[List[str]]):
        """
        Construye vocabulario desde una lista de frases tokenizadas.
        
        Args:
            sentences: Lista de frases, cada una como lista de tokens
        """
        # Contar frecuencias
        for sentence in sentences:
            self.token_freqs.update(sentence)
        
        # Añadir tokens que superen el umbral
        idx = len(self.token2idx)
        for token, freq in self.token_freqs.items():
            if freq >= self.freq_threshold and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
        
        logger.info(f"Vocabulario construido con {len(self.token2idx)} tokens")
    
    def encode(self, sentence: List[str]) -> List[int]:
        """
        Convierte tokens a índices.
        
        Args:
            sentence: Lista de tokens
            
        Returns:
            Lista de índices
        """
        return [self.token2idx.get(token, self.token2idx[self.unk_token]) for token in sentence]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        Convierte índices a tokens.
        
        Args:
            indices: Lista de índices
            
        Returns:
            Lista de tokens
        """
        return [self.idx2token.get(idx, self.unk_token) for idx in indices]
    
    def __len__(self):
        return len(self.token2idx)
    
    @property
    def pad_idx(self):
        return self.token2idx[self.pad_token]
    
    @property
    def sos_idx(self):
        return self.token2idx[self.sos_token]
    
    @property
    def eos_idx(self):
        return self.token2idx[self.eos_token]
    
    @property
    def unk_idx(self):
        return self.token2idx[self.unk_token]
    
    def save(self, path: str):
        """Guarda vocabulario en disco."""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'token_freqs': self.token_freqs,
                'freq_threshold': self.freq_threshold
            }, f)
        logger.info(f"Vocabulario guardado en {path}")
    
    @classmethod
    def load(cls, path: str):
        """Carga vocabulario desde disco."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(freq_threshold=data['freq_threshold'])
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        vocab.token_freqs = data['token_freqs']
        
        logger.info(f"Vocabulario cargado desde {path}")
        return vocab


class TranslationDataset(Dataset):
    """
    Dataset para pares de traducción.
    """
    
    def __init__(
        self,
        src_sentences: List[List[str]],
        tgt_sentences: List[List[str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_length: int = 128
    ):
        """
        Args:
            src_sentences: Frases fuente (tokenizadas)
            tgt_sentences: Frases objetivo (tokenizadas)
            src_vocab: Vocabulario fuente
            tgt_vocab: Vocabulario objetivo
            max_length: Longitud máxima de secuencia
        """
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # Filtrar secuencias muy largas
        filtered_pairs = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            if len(src) <= max_length and len(tgt) <= max_length:
                filtered_pairs.append((src, tgt))
        
        self.src_sentences, self.tgt_sentences = zip(*filtered_pairs) if filtered_pairs else ([], [])
        logger.info(f"Dataset creado con {len(self)} pares")
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna un par source-target codificado.
        
        Returns:
            src_tensor: Índices de tokens source
            tgt_tensor: Índices de tokens target
        """
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        
        # Codificar
        src_indices = [self.src_vocab.sos_idx] + self.src_vocab.encode(src) + [self.src_vocab.eos_idx]
        tgt_indices = [self.tgt_vocab.sos_idx] + self.tgt_vocab.encode(tgt) + [self.tgt_vocab.eos_idx]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 0):
    """
    Función de collate para DataLoader.
    
    Agrupa múltiples ejemplos en un batch, aplicando padding.
    
    Args:
        batch: Lista de tuplas (src_tensor, tgt_tensor)
        pad_idx: Índice del token de padding
        
    Returns:
        src_batch: (batch_size, max_src_len)
        tgt_batch: (batch_size, max_tgt_len)
    """
    src_batch, tgt_batch = [], []
    
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)
    
    # Padding
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    return src_batch, tgt_batch


def simple_tokenize(text: str, lang: str = 'es') -> List[str]:
    """
    Tokenización simple basada en espacios y puntuación.
    
    Args:
        text: Texto a tokenizar
        lang: Idioma ('es' o 'en')
        
    Returns:
        Lista de tokens
    """
    import re
    
    # Normalizar
    text = text.lower().strip()
    
    # Separar puntuación
    text = re.sub(r"([.!?¿¡,;:])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    
    # Split
    tokens = text.split()
    
    return tokens


def download_tatoeba(data_dir: str = "data/raw") -> Tuple[str, str]:
    """
    Descarga el dataset Tatoeba español-inglés.
    
    Args:
        data_dir: Directorio donde guardar los datos
        
    Returns:
        Tuplas con rutas a archivos español e inglés
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Intentar con Hugging Face datasets
    try:
        from datasets import load_dataset
        
        logger.info("Descargando Tatoeba desde Hugging Face...")
        ds = load_dataset("Helsinki-NLP/tatoeba_mt", "spa-eng", split="test")
        
        # Limitar a 20k muestras
        max_samples = min(20000, len(ds))
        ds = ds.select(range(max_samples))
        
        # Guardar en archivos
        es_path = os.path.join(data_dir, "tatoeba.es")
        en_path = os.path.join(data_dir, "tatoeba.en")
        
        with open(es_path, 'w', encoding='utf-8') as f_es, \
             open(en_path, 'w', encoding='utf-8') as f_en:
            for example in ds:
                # Estructura puede variar, intentar diferentes campos
                src = example.get('sourceString') or example.get('translation', {}).get('es', '')
                tgt = example.get('targetString') or example.get('translation', {}).get('en', '')
                
                if src and tgt:
                    f_es.write(src.strip() + '\n')
                    f_en.write(tgt.strip() + '\n')
        
        logger.info(f"Dataset guardado en {es_path} y {en_path}")
        return es_path, en_path
        
    except Exception as e:
        logger.warning(f"Error descargando con Hugging Face: {e}")
        logger.info("Creando dataset de ejemplo...")
        
        # Dataset de ejemplo pequeño
        example_pairs = [
            ("hola mundo", "hello world"),
            ("buenos días", "good morning"),
            ("¿cómo estás?", "how are you?"),
            ("gracias", "thank you"),
            ("de nada", "you're welcome"),
            ("adiós", "goodbye"),
            ("por favor", "please"),
            ("lo siento", "i'm sorry"),
            ("te quiero", "i love you"),
            ("hasta luego", "see you later"),
        ] * 100  # Repetir para tener más datos
        
        es_path = os.path.join(data_dir, "tatoeba.es")
        en_path = os.path.join(data_dir, "tatoeba.en")
        
        with open(es_path, 'w', encoding='utf-8') as f_es, \
             open(en_path, 'w', encoding='utf-8') as f_en:
            for es, en in example_pairs:
                f_es.write(es + '\n')
                f_en.write(en + '\n')
        
        logger.info(f"Dataset de ejemplo creado con {len(example_pairs)} pares")
        return es_path, en_path


def load_data(
    data_dir: str = "data/raw",
    max_samples: int = 20000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Dict[str, Tuple[List[List[str]], List[List[str]]]]:
    """
    Carga y divide los datos en train/val/test.
    
    Args:
        data_dir: Directorio de datos
        max_samples: Máximo de muestras a cargar
        train_split: Fracción para entrenamiento
        val_split: Fracción para validación
        test_split: Fracción para test
        
    Returns:
        Diccionario con splits {'train': (src, tgt), 'val': (src, tgt), 'test': (src, tgt)}
    """
    # Descargar si no existe
    es_path = os.path.join(data_dir, "tatoeba.es")
    en_path = os.path.join(data_dir, "tatoeba.en")
    
    if not os.path.exists(es_path) or not os.path.exists(en_path):
        es_path, en_path = download_tatoeba(data_dir)
    
    # Leer archivos
    with open(es_path, 'r', encoding='utf-8') as f:
        es_lines = [line.strip() for line in f if line.strip()]
    
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f if line.strip()]
    
    # Asegurar que tienen el mismo tamaño
    assert len(es_lines) == len(en_lines), "Archivos desbalanceados"
    
    # Limitar muestras
    if len(es_lines) > max_samples:
        es_lines = es_lines[:max_samples]
        en_lines = en_lines[:max_samples]
    
    # Tokenizar
    logger.info("Tokenizando...")
    es_tokenized = [simple_tokenize(line, 'es') for line in es_lines]
    en_tokenized = [simple_tokenize(line, 'en') for line in en_lines]
    
    # Dividir en splits
    n = len(es_tokenized)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    
    splits = {
        'train': (es_tokenized[:train_end], en_tokenized[:train_end]),
        'val': (es_tokenized[train_end:val_end], en_tokenized[train_end:val_end]),
        'test': (es_tokenized[val_end:], en_tokenized[val_end:])
    }
    
    logger.info(f"Datos cargados - Train: {len(splits['train'][0])}, Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")
    
    return splits


def create_dataloaders(
    splits: Dict[str, Tuple[List[List[str]], List[List[str]]]],
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """
    Crea DataLoaders para cada split.
    
    Args:
        splits: Diccionario con splits de datos
        src_vocab: Vocabulario fuente
        tgt_vocab: Vocabulario objetivo
        batch_size: Tamaño del batch
        max_length: Longitud máxima de secuencia
        num_workers: Número de workers para DataLoader
        
    Returns:
        Diccionario con DataLoaders
    """
    dataloaders = {}
    
    for split_name, (src_sentences, tgt_sentences) in splits.items():
        dataset = TranslationDataset(
            src_sentences, tgt_sentences,
            src_vocab, tgt_vocab,
            max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            collate_fn=lambda batch: collate_fn(batch, src_vocab.pad_idx),
            pin_memory=True
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders


def prepare_data(config: dict, force_download: bool = False):
    """
    Pipeline completo de preparación de datos.
    
    Args:
        config: Configuración del proyecto
        force_download: Si True, re-descarga los datos
    """
    data_config = config.get('data', {})
    data_dir = "data/raw"
    processed_dir = "data/processed"
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Descargar datos
    logger.info("Descargando datos...")
    if force_download or not os.path.exists(os.path.join(data_dir, "tatoeba.es")):
        download_tatoeba(data_dir)
    
    # Cargar y dividir
    logger.info("Cargando datos...")
    splits = load_data(
        data_dir,
        max_samples=data_config.get('max_samples', 20000),
        train_split=data_config.get('train_split', 0.8),
        val_split=data_config.get('val_split', 0.1),
        test_split=data_config.get('test_split', 0.1)
    )
    
    # Construir vocabularios
    logger.info("Construyendo vocabularios...")
    src_vocab = Vocabulary(freq_threshold=2)
    tgt_vocab = Vocabulary(freq_threshold=2)
    
    src_vocab.build_vocabulary(splits['train'][0])
    tgt_vocab.build_vocabulary(splits['train'][1])
    
    # Guardar vocabularios
    src_vocab.save(os.path.join(processed_dir, "vocab_src.pkl"))
    tgt_vocab.save(os.path.join(processed_dir, "vocab_tgt.pkl"))
    
    logger.info(f"Vocabulario español: {len(src_vocab)} tokens")
    logger.info(f"Vocabulario inglés: {len(tgt_vocab)} tokens")
    logger.info("Preparación de datos completada")


if __name__ == "__main__":
    import argparse
    import yaml
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(description="Preparar datos para NMT")
    parser.add_argument("--prepare", action="store_true", help="Preparar datos")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Ruta al archivo de configuración")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.prepare:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        prepare_data(config)
