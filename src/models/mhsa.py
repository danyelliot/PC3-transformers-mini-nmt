"""
Multi-Head Self-Attention y Multi-Head Cross-Attention.

Implementa la arquitectura completa de atención multi-cabeza con
proyecciones lineales para Q, K, V y salida.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention según "Attention is All You Need".
    
    Divide d_model en n_heads cabezas, cada una con dimensión d_k = d_model / n_heads.
    Cada cabeza aprende diferentes patrones de atención.
    
    Args:
        d_model: Dimensión del modelo
        n_heads: Número de cabezas
        dropout: Tasa de dropout
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Proyecciones lineales para Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Proyección de salida
        self.out_linear = nn.Linear(d_model, d_model)
        # Atención scaled dot-product
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Positional helpers (pueden ser inyectados desde Transformer)
        self.rotary = None  # tipo: Optional[RotaryPositionalEmbedding]
        self.alibi = None   # tipo: Optional[ALiBiPositionalBias]

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
        Forward pass de Multi-Head Attention.
        
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model) donde seq_len_k == seq_len_v
            mask: Máscara de atención (batch, 1, seq_len_q, seq_len_k) o broadcast compatible
            return_attention: Si True, retorna también los pesos de atención
            
        Returns:
            output: (batch, seq_len_q, d_model)
            attn_weights: (batch, n_heads, seq_len_q, seq_len_k) si return_attention=True
        """
        batch_size = query.size(0)
        
        # Proyecciones lineales: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Dividir en múltiples cabezas
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, head_dim)
        # -> (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Aplicar atención
        # Aplicar rotatory positional embedding si está disponible
        if self.rotary is not None:
            try:
                Q, K = self.rotary(Q, K)
            except Exception:
                # en caso de que rotary espere shapes distintas, intentar con seq_len
                seq_len_q = Q.size(2)
                Q, K = self.rotary(Q, K, seq_len=seq_len_q)

        # Preparar bias (ALiBi) si existe
        bias = None
        if self.alibi is not None:
            # alibi.forward espera seq_len y retorna (1, n_heads, seq_len, seq_len)
            seq_len_q = Q.size(2)
            seq_len_k = K.size(2)
            # Si shapes difieren (cross-attention), recomputar bias apropiado
            if seq_len_q == seq_len_k:
                bias = self.alibi(seq_len_q)
            else:
                # generar bias para la mayor de las longitudes y slice según necesidad
                max_len = max(seq_len_q, seq_len_k)
                full = self.alibi(max_len)
                bias = full[:, :, :seq_len_q, :seq_len_k]

        # output: (batch, n_heads, seq_len_q, head_dim)
        # attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
        attn_output, attn_weights = self.attention(Q, K, V, mask, bias=bias, return_attention=return_attention)
        
        # Concatenar cabezas
        # (batch, n_heads, seq_len_q, head_dim) -> (batch, seq_len_q, n_heads, head_dim)
        # -> (batch, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Proyección final
        output = self.out_linear(attn_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class MultiHeadSelfAttention(MultiHeadAttention):
    """
    Multi-Head Self-Attention (Q, K, V vienen de la misma fuente).
    """
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward para self-attention.
        
        Args:
            x: (batch, seq_len, d_model)
            mask: Máscara de atención
            return_attention: Si retornar pesos de atención
            
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: opcional
        """
        return super().forward(x, x, x, mask, return_attention)


class MultiHeadCrossAttention(MultiHeadAttention):
    """
    Multi-Head Cross-Attention (Q viene del decoder, K y V del encoder).
    
    Se usa en el decoder para atender a las salidas del encoder.
    """
    
    def forward(
        self,
        query: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward para cross-attention.
        
        Args:
            query: Salida del decoder (batch, tgt_len, d_model)
            encoder_output: Salida del encoder (batch, src_len, d_model)
            mask: Máscara (típicamente para padding del source)
            return_attention: Si retornar pesos de atención
            
        Returns:
            output: (batch, tgt_len, d_model)
            attn_weights: opcional
        """
        # Q viene del decoder, K y V del encoder
        return super().forward(query, encoder_output, encoder_output, mask, return_attention)


class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network (FFN) del Transformer.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Típicamente, dim_feedforward = 4 * d_model
    
    Args:
        d_model: Dimensión del modelo
        dim_feedforward: Dimensión de la capa intermedia
        dropout: Tasa de dropout
    """
    
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, dim_feedforward)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # (batch, seq_len, dim_feedforward) -> (batch, seq_len, d_model)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


def test_mhsa_shapes():
    """Test: verificar que los shapes sean correctos en MHSA."""
    batch_size = 4
    seq_len = 32
    d_model = 256
    n_heads = 8
    
    mhsa = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, _ = mhsa(x)
    
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Shape incorrecto: esperado {(batch_size, seq_len, d_model)}, obtenido {output.shape}"
    
    print("✓ Test MHSA: shapes correctos")


def test_cross_attention_shapes():
    """Test: verificar shapes en cross-attention."""
    batch_size = 4
    src_len = 20
    tgt_len = 15
    d_model = 256
    n_heads = 8
    
    cross_attn = MultiHeadCrossAttention(d_model=d_model, n_heads=n_heads)
    
    decoder_output = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    output, attn_weights = cross_attn(decoder_output, encoder_output, return_attention=True)
    
    assert output.shape == (batch_size, tgt_len, d_model)
    assert attn_weights.shape == (batch_size, n_heads, tgt_len, src_len)
    
    print("✓ Test Cross-Attention: shapes correctos")


def test_ffn():
    """Test: verificar FFN."""
    batch_size = 4
    seq_len = 32
    d_model = 256
    dim_feedforward = 1024
    
    ffn = FeedForwardNetwork(d_model=d_model, dim_feedforward=dim_feedforward)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    
    print("✓ Test FFN: correcto")


if __name__ == "__main__":
    # Ejecutar tests
    test_mhsa_shapes()
    test_cross_attention_shapes()
    test_ffn()
    print("\n✅ Todos los tests de MHSA pasaron!")
