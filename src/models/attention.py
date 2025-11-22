"""
Implementación de Scaled Dot-Product Attention con máscaras y KV-cache.

Este módulo incluye:
- Scaled Dot-Product Attention manual
- Máscaras: causal, padding, selectiva
- KV-cache para generación eficiente
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention según el paper "Attention is All You Need".
    
    Fórmula: Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    
    Args:
        dropout: Tasa de dropout (default: 0.1)
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass de la atención.
        
        Args:
            query: (batch, n_heads, seq_len_q, d_k)
            key: (batch, n_heads, seq_len_k, d_k)
            value: (batch, n_heads, seq_len_v, d_v)  donde seq_len_k == seq_len_v
            mask: (batch, 1, seq_len_q, seq_len_k) o broadcast compatible
                  True/1 = mantener, False/0 = enmascarar
            return_attention: Si True, retorna también los pesos de atención
            
        Returns:
            output: (batch, n_heads, seq_len_q, d_v)
            attn_weights: (batch, n_heads, seq_len_q, seq_len_k) si return_attention=True
        """
        d_k = query.size(-1)
        
        # Q·K^T / √d_k
        # (batch, n_heads, seq_len_q, d_k) @ (batch, n_heads, d_k, seq_len_k)
        # -> (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Aplicar máscara (poner -inf donde mask == 0 o False)
        if mask is not None:
            # Si la máscara es booleana, convertir: True -> 0, False -> -inf
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                # Si es float, asumir que 0 = enmascarar
                scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax sobre la última dimensión (seq_len_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout en los pesos de atención
        attn_weights = self.dropout(attn_weights)
        
        # Atención · V
        # (batch, n_heads, seq_len_q, seq_len_k) @ (batch, n_heads, seq_len_v, d_v)
        # -> (batch, n_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, value)
        
        if return_attention:
            return output, attn_weights
        return output, None


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Crea máscara causal (triangular inferior) para atención autoregresiva.
    
    El token en posición i solo puede atender a tokens en posiciones j <= i.
    
    Args:
        seq_len: Longitud de la secuencia
        device: Dispositivo donde crear el tensor
        
    Returns:
        Máscara booleana (seq_len, seq_len) donde True = permitido, False = bloqueado
        
    Ejemplo:
        seq_len = 4
        [[True, False, False, False],
         [True, True,  False, False],
         [True, True,  True,  False],
         [True, True,  True,  True ]]
    """
    # Crear matriz triangular inferior (incluye diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def create_padding_mask(
    seq: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Crea máscara de padding para secuencias con diferentes longitudes.
    
    Args:
        seq: Tensor de tokens (batch, seq_len)
        pad_idx: Índice del token de padding (default: 0)
        
    Returns:
        Máscara booleana (batch, 1, 1, seq_len) donde True = válido, False = padding
        
    Ejemplo:
        seq = [[1, 2, 3, 0, 0],
               [4, 5, 0, 0, 0]]
        ->
        [[[[ True,  True,  True, False, False]],
          [[ True,  True, False, False, False]]]]
    """
    # seq != pad_idx -> True si es un token válido
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    return mask


def create_cross_attention_mask(
    src_seq: torch.Tensor,
    tgt_seq: torch.Tensor,
    src_pad_idx: int = 0,
    tgt_pad_idx: int = 0
) -> torch.Tensor:
    """
    Crea máscara para cross-attention (decoder -> encoder).
    
    Args:
        src_seq: Secuencia source (batch, src_len)
        tgt_seq: Secuencia target (batch, tgt_len)
        src_pad_idx: Índice de padding en source
        tgt_pad_idx: Índice de padding en target
        
    Returns:
        Máscara (batch, 1, tgt_len, src_len)
    """
    # Máscara de padding del source (batch, 1, 1, src_len)
    src_mask = create_padding_mask(src_seq, src_pad_idx)
    
    # Expandir para que coincida con tgt_len
    # (batch, 1, 1, src_len) -> (batch, 1, tgt_len, src_len)
    tgt_len = tgt_seq.size(1)
    cross_mask = src_mask.expand(-1, -1, tgt_len, -1)
    
    return cross_mask


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """
    Combina múltiples máscaras usando AND lógico.
    
    Args:
        *masks: Máscaras a combinar
        
    Returns:
        Máscara combinada (True solo si todas las máscaras son True)
    """
    if not masks:
        return None
    
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined & mask
    
    return combined


class KVCache:
    """
    Cache para almacenar Keys y Values durante la generación autoregresiva.
    
    Reduce la complejidad de O(n²) a O(n) al no recalcular atención 
    para tokens ya generados.
    """
    
    def __init__(self, max_batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: torch.device):
        """
        Args:
            max_batch_size: Tamaño máximo del batch
            max_seq_len: Longitud máxima de secuencia a cachear
            n_heads: Número de cabezas de atención
            head_dim: Dimensión de cada cabeza
            device: Dispositivo donde almacenar el cache
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        # Pre-asignar memoria
        self.cache_k = torch.zeros(
            (max_batch_size, n_heads, max_seq_len, head_dim),
            dtype=torch.float32,
            device=device
        )
        self.cache_v = torch.zeros(
            (max_batch_size, n_heads, max_seq_len, head_dim),
            dtype=torch.float32,
            device=device
        )
        
        self.current_length = 0
        
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Actualiza el cache con nuevos keys y values.
        
        Args:
            key: (batch, n_heads, seq_len, head_dim)
            value: (batch, n_heads, seq_len, head_dim)
            start_pos: Posición de inicio (si None, usa current_length)
            
        Returns:
            Keys y values completos hasta la posición actual
        """
        if start_pos is None:
            start_pos = self.current_length
        
        batch_size, n_heads, seq_len, head_dim = key.shape
        
        # Verificar límites
        assert batch_size <= self.max_batch_size
        assert start_pos + seq_len <= self.max_seq_len
        
        # Guardar en el cache
        self.cache_k[:batch_size, :, start_pos:start_pos + seq_len, :] = key
        self.cache_v[:batch_size, :, start_pos:start_pos + seq_len, :] = value
        
        self.current_length = start_pos + seq_len
        
        # Retornar todo el cache hasta la posición actual
        return (
            self.cache_k[:batch_size, :, :self.current_length, :],
            self.cache_v[:batch_size, :, :self.current_length, :]
        )
    
    def reset(self):
        """Resetea el cache."""
        self.current_length = 0
        self.cache_k.zero_()
        self.cache_v.zero_()
    
    def get_current_length(self) -> int:
        """Retorna la longitud actual del cache."""
        return self.current_length


class PagedKVCache:
    """
    KV-cache paginado que divide el cache en bloques de tamaño fijo.
    
    Permite mover bloques viejos a CPU cuando se llena la memoria,
    útil para contextos muy largos.
    """
    
    def __init__(
        self,
        max_batch_size: int,
        n_heads: int,
        head_dim: int,
        block_size: int = 256,
        max_blocks: int = 32,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            max_batch_size: Tamaño máximo del batch
            n_heads: Número de cabezas
            head_dim: Dimensión de cada cabeza
            block_size: Tamaño de cada bloque (default: 256 tokens)
            max_blocks: Número máximo de bloques activos
            device: Dispositivo principal
        """
        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device
        
        # Lista de bloques activos
        self.blocks_k = []
        self.blocks_v = []
        
        self.current_block_idx = 0
        self.current_position_in_block = 0
        
    def update(self, key: torch.Tensor, value: torch.Tensor):
        """Actualiza el cache paginado."""
        batch_size, n_heads, seq_len, head_dim = key.shape
        
        # Si necesitamos un nuevo bloque
        if self.current_position_in_block + seq_len > self.block_size:
            self._allocate_new_block()
        
        # Guardar en el bloque actual
        start = self.current_position_in_block
        end = start + seq_len
        
        if self.current_block_idx >= len(self.blocks_k):
            self._allocate_new_block()
        
        self.blocks_k[self.current_block_idx][:batch_size, :, start:end, :] = key
        self.blocks_v[self.current_block_idx][:batch_size, :, start:end, :] = value
        
        self.current_position_in_block += seq_len
        
        return self.get_all()
    
    def _allocate_new_block(self):
        """Asigna un nuevo bloque de memoria."""
        new_block_k = torch.zeros(
            (self.max_batch_size, self.n_heads, self.block_size, self.head_dim),
            device=self.device
        )
        new_block_v = torch.zeros(
            (self.max_batch_size, self.n_heads, self.block_size, self.head_dim),
            device=self.device
        )
        
        self.blocks_k.append(new_block_k)
        self.blocks_v.append(new_block_v)
        
        self.current_block_idx += 1
        self.current_position_in_block = 0
        
        # Si excedemos max_blocks, mover bloques viejos a CPU
        if len(self.blocks_k) > self.max_blocks:
            old_block_k = self.blocks_k.pop(0).cpu()
            old_block_v = self.blocks_v.pop(0).cpu()
            # En una implementación completa, guardaríamos estos en disco
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna todos los keys y values concatenados."""
        if not self.blocks_k:
            return None, None
        
        all_k = torch.cat(self.blocks_k, dim=2)
        all_v = torch.cat(self.blocks_v, dim=2)
        
        return all_k, all_v
    
    def reset(self):
        """Resetea el cache."""
        self.blocks_k = []
        self.blocks_v = []
        self.current_block_idx = 0
        self.current_position_in_block = 0
