"""
Codificaciones Posicionales para Transformers.

Implementaciones:
- Sinusoidal (original de "Attention is All You Need")
- RoPE (Rotary Position Embedding) con interpolación
- ALiBi (Attention with Linear Biases)
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal del paper original.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Dimensión del modelo
        max_len: Longitud máxima de secuencia
        dropout: Tasa de dropout
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Crear matriz de codificación posicional
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calcular div_term = 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Aplicar sin a posiciones pares, cos a impares
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Añadir dimensión de batch: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Registrar como buffer (no es un parámetro entrenable)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            x + positional encoding
        """
        # x.shape: (batch, seq_len, d_model)
        # self.pe.shape: (1, max_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) - https://arxiv.org/abs/2104.09864
    
    En vez de sumar un vector posicional, rota las queries y keys según la posición.
    Esto preserva la información relativa de posición en el producto punto.
    
    Args:
        dim: Dimensión de cada cabeza
        max_seq_len: Longitud máxima de secuencia
        base: Base para la frecuencia (default: 10000)
        scaling_factor: Factor de escalado para interpolación (default: 1.0)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Pre-computar frecuencias
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-computar cos y sin para max_seq_len
        self._compute_cos_sin_cache(max_seq_len)
    
    def _compute_cos_sin_cache(self, seq_len: int):
        """Pre-computa cache de cos y sin para eficiencia."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        t = t / self.scaling_factor  # Aplicar scaling
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)

        # Guardar cos y sin con tamaño (1,1,seq_len, dim/2) para multiplicar con mitades de la cabeza
        self.register_buffer('cos_cached', freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', freqs.sin()[None, None, :, :], persistent=False)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica RoPE a queries y keys.
        
        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_heads, seq_len, head_dim)
            seq_len: Longitud de la secuencia (si None, se infiere de q)
            
        Returns:
            q_rotated, k_rotated con las mismas dimensiones
        """
        if seq_len is None:
            seq_len = q.size(2)
        
        # Si la secuencia es más larga que el cache, recomputar
        if seq_len > self.cos_cached.size(2):
            self._compute_cos_sin_cache(seq_len)
        
        # Obtener cos y sin para esta longitud
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Aplicar rotación
        q_rotated = self._apply_rotary_emb(q, cos, sin)
        k_rotated = self._apply_rotary_emb(k, cos, sin)
        
        return q_rotated, k_rotated
    
    @staticmethod
    def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotación de RoPE.
        
        La rotación se hace dividiendo x en dos mitades y aplicando:
        x_rotated = [x1*cos - x2*sin, x1*sin + x2*cos]
        """
        # Dividir en dos mitades
        x1, x2 = x.chunk(2, dim=-1)
        
        # Aplicar rotación
        # cos y sin ya tienen la forma correcta (1, 1, seq_len, dim)
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return x_rotated
    
    def apply_ntk_scaling(self, target_seq_len: int, alpha: float = 1.0):
        """
        Aplica NTK-aware scaling para extrapolar a longitudes mayores.
        
        Ajusta dinámicamente la base de frecuencias según:
        base_new = base * ((alpha * target_len / original_len) ^ (dim / (dim - 2)))
        
        Args:
            target_seq_len: Longitud objetivo
            alpha: Factor de escala (típicamente 1.0)
        """
        if target_seq_len <= self.max_seq_len:
            return
        
        # Calcular nuevo base
        scaling_ratio = target_seq_len / self.max_seq_len
        new_base = self.base * (alpha * scaling_ratio) ** (self.dim / (self.dim - 2))
        
        # Recomputar frecuencias
        inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq.to(self.inv_freq.device))
        
        # Recomputar cache
        self._compute_cos_sin_cache(target_seq_len)
        self.max_seq_len = target_seq_len


class ALiBiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) - https://arxiv.org/abs/2108.12409
    
    En vez de codificaciones posicionales, añade un sesgo lineal directamente
    a los scores de atención: bias = -m * |i - j|
    
    Cada cabeza tiene su propia pendiente m.
    
    Args:
        n_heads: Número de cabezas de atención
        max_seq_len: Longitud máxima de secuencia
    """
    
    def __init__(self, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Calcular pendientes para cada cabeza
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
        
        # Pre-computar la matriz de distancias relativas
        self._compute_bias_cache(max_seq_len)
    
    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        """
        Calcula las pendientes para cada cabeza.
        
        Para n_heads que es potencia de 2:
        m_i = 2^(-8i/n_heads) para i en [1, n_heads]
        
        Returns:
            Tensor (n_heads,) con las pendientes
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return torch.tensor([start * (ratio ** i) for i in range(n)])
        
        # Si n_heads es potencia de 2
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        
        # Si no, usar la potencia de 2 más cercana y extrapolar
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        
        # Extrapolar para el resto
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
        slopes = torch.cat([slopes, extra_slopes])
        
        return slopes
    
    def _compute_bias_cache(self, seq_len: int):
        """
        Pre-computa la matriz de sesgos.
        
        Para cada posición (i, j), el sesgo es -slope * |i - j|
        """
        # Crear matriz de distancias relativas
        # (seq_len, seq_len) donde entry[i,j] = i - j
        context_position = torch.arange(seq_len)[:, None]
        memory_position = torch.arange(seq_len)[None, :]
        relative_position = memory_position - context_position
        
        # Tomar valor absoluto y multiplicar por -1
        relative_position = -torch.abs(relative_position).float()
        
        # Expandir para cada cabeza y multiplicar por slopes
        # (1, n_heads, seq_len, seq_len)
        alibi = relative_position.unsqueeze(0).unsqueeze(0)
        alibi = alibi * self.slopes.view(1, -1, 1, 1)
        
        self.register_buffer('alibi_bias', alibi, persistent=False)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Retorna el sesgo de ALiBi para una longitud dada.
        
        Args:
            seq_len: Longitud de la secuencia
            
        Returns:
            Sesgo (1, n_heads, seq_len, seq_len)
        """
        # Si seq_len es mayor que el cache, recomputar
        if seq_len > self.alibi_bias.size(2):
            self._compute_bias_cache(seq_len)
        
        # Retornar la porción necesaria
        return self.alibi_bias[:, :, :seq_len, :seq_len]


def test_rope_norm_preservation():
    """
    Test: RoPE debe preservar la norma de Q y K.
    """
    batch, n_heads, seq_len, head_dim = 2, 8, 64, 32
    
    q = torch.randn(batch, n_heads, seq_len, head_dim)
    k = torch.randn(batch, n_heads, seq_len, head_dim)
    
    rope = RotaryPositionalEmbedding(dim=head_dim)
    q_rot, k_rot = rope(q, k)
    
    # Verificar que la norma se preserva
    q_norm_before = q.norm(dim=-1)
    q_norm_after = q_rot.norm(dim=-1)
    
    assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5), \
        "RoPE debe preservar la norma de Q"
    
    print("✓ Test RoPE: norma preservada")


def test_alibi_monotonicity():
    """
    Test: ALiBi debe ser monotónicamente decreciente con la distancia.
    """
    n_heads = 8
    seq_len = 100
    
    alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=seq_len)
    bias = alibi(seq_len)
    
    # Para la primera cabeza, primera posición
    # bias[0, 0, 0, :] debe ser monotónicamente decreciente
    head_0_pos_0 = bias[0, 0, 0, :]
    
    # Verificar monotonicidad: cada valor debe ser <= al anterior
    diffs = head_0_pos_0[1:] - head_0_pos_0[:-1]
    assert (diffs <= 0).all(), "ALiBi debe ser monotónicamente decreciente"
    
    print("✓ Test ALiBi: monotonicidad verificada")


if __name__ == "__main__":
    # Ejecutar tests
    test_rope_norm_preservation()
    test_alibi_monotonicity()
    print("\n✅ Todos los tests de codificaciones posicionales pasaron!")
