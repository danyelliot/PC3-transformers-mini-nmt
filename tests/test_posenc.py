"""
Tests para codificaciones posicionales.
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.posenc import (
    SinusoidalPositionalEncoding,
    RotaryPositionalEmbedding,
    ALiBiPositionalBias
)


class TestSinusoidalPositionalEncoding:
    """Tests para codificación posicional sinusoidal."""
    
    def test_output_shape(self):
        """Verifica que el output tenga el mismo shape que el input."""
        batch_size = 4
        seq_len = 32
        d_model = 256
        
        pos_enc = SinusoidalPositionalEncoding(d_model, max_len=512, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == x.shape
    
    def test_deterministic(self):
        """Verifica que la codificación sea determinista."""
        d_model = 128
        seq_len = 50
        
        pos_enc = SinusoidalPositionalEncoding(d_model, max_len=512, dropout=0.0)
        
        x1 = torch.randn(2, seq_len, d_model)
        x2 = x1.clone()
        
        out1 = pos_enc(x1)
        out2 = pos_enc(x2)
        
        # Los cambios deben ser idénticos
        assert torch.allclose(out1, out2)


class TestRotaryPositionalEmbedding:
    """Tests para RoPE."""
    
    def test_norm_preservation(self):
        """
        Test crítico: RoPE debe preservar la norma de Q y K.
        """
        batch_size = 2
        n_heads = 8
        seq_len = 64
        head_dim = 32
        
        rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=128)
        
        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        # Normas antes de aplicar RoPE
        q_norm_before = q.norm(dim=-1)
        k_norm_before = k.norm(dim=-1)
        
        # Aplicar RoPE
        q_rot, k_rot = rope(q, k)
        
        # Normas después de RoPE
        q_norm_after = q_rot.norm(dim=-1)
        k_norm_after = k_rot.norm(dim=-1)
        
        # Verificar preservación de norma (tolerancia 1e-5)
        assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5), \
            "RoPE debe preservar la norma de Q"
        assert torch.allclose(k_norm_before, k_norm_after, atol=1e-5), \
            "RoPE debe preservar la norma de K"
    
    def test_output_shape(self):
        """Verifica que los shapes no cambien."""
        batch_size = 4
        n_heads = 8
        seq_len = 32
        head_dim = 64
        
        rope = RotaryPositionalEmbedding(dim=head_dim)
        
        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_longer_sequences(self):
        """Verifica que RoPE maneje secuencias más largas que max_seq_len."""
        head_dim = 32
        rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=64)
        
        # Secuencia más larga que max_seq_len
        q = torch.randn(1, 1, 128, head_dim)
        k = torch.randn(1, 1, 128, head_dim)
        
        # No debe lanzar error
        q_rot, k_rot = rope(q, k, seq_len=128)
        
        assert q_rot.shape == q.shape
    
    def test_ntk_scaling(self):
        """Verifica que NTK scaling funcione sin errores."""
        rope = RotaryPositionalEmbedding(dim=32, max_seq_len=512)
        
        # Aplicar NTK scaling para secuencias más largas
        rope.apply_ntk_scaling(target_seq_len=2048, alpha=1.0)
        
        # Verificar que max_seq_len se actualizó
        assert rope.max_seq_len == 2048


class TestALiBiPositionalBias:
    """Tests para ALiBi."""
    
    def test_monotonicity(self):
        """
        Test crítico: ALiBi debe ser monotónicamente decreciente con la distancia.
        """
        n_heads = 8
        seq_len = 100
        
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=seq_len)
        bias = alibi(seq_len)
        
        # bias shape: (1, n_heads, seq_len, seq_len)
        # Para cada cabeza, verificar monotonicidad
        
        for head_idx in range(n_heads):
            # Para la posición 0, mirar todos los offsets
            head_bias = bias[0, head_idx, 0, :]  # (seq_len,)
            
            # El sesgo debe ser 0 en la diagonal (distancia 0)
            assert head_bias[0] == 0.0
            
            # Verificar que es monotónicamente decreciente
            for i in range(seq_len - 1):
                assert head_bias[i] >= head_bias[i + 1], \
                    f"ALiBi debe ser monotónicamente decreciente. " \
                    f"Head {head_idx}, pos {i}: {head_bias[i]:.4f} >= {head_bias[i+1]:.4f}"
    
    def test_output_shape(self):
        """Verifica el shape del output."""
        n_heads = 4
        seq_len = 50
        
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=256)
        bias = alibi(seq_len)
        
        assert bias.shape == (1, n_heads, seq_len, seq_len)
    
    def test_different_slopes_per_head(self):
        """Verifica que cada cabeza tenga una pendiente diferente."""
        n_heads = 8
        seq_len = 10
        
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=128)
        bias = alibi(seq_len)
        
        # Extraer pendientes implícitas
        slopes = []
        for head_idx in range(n_heads):
            # Pendiente = bias[posición 0, offset 1]
            slope = bias[0, head_idx, 0, 1].item()
            slopes.append(slope)
        
        # Verificar que no todas las pendientes son iguales
        assert len(set(slopes)) > 1, "Las cabezas deben tener pendientes diferentes"
    
    def test_negative_bias(self):
        """Verifica que el sesgo sea siempre <= 0."""
        n_heads = 4
        seq_len = 20
        
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=128)
        bias = alibi(seq_len)
        
        # Todos los valores deben ser <= 0
        assert (bias <= 0).all(), "ALiBi bias debe ser siempre <= 0"
    
    def test_longer_sequences(self):
        """Verifica manejo de secuencias más largas."""
        alibi = ALiBiPositionalBias(n_heads=8, max_seq_len=64)
        
        # Secuencia más larga
        bias = alibi(128)
        
        assert bias.shape == (1, 8, 128, 128)


class TestPositionalEncodingComparison:
    """Tests comparativos entre diferentes codificaciones."""
    
    def test_all_encodings_same_device(self):
        """Verifica que todas funcionen en el mismo device."""
        d_model = 128
        head_dim = d_model // 8
        n_heads = 8
        seq_len = 32
        
        device = torch.device('cpu')
        
        # Sinusoidal
        sin_enc = SinusoidalPositionalEncoding(d_model, dropout=0.0)
        x = torch.randn(2, seq_len, d_model, device=device)
        out_sin = sin_enc(x)
        assert out_sin.device == device
        
        # RoPE
        rope = RotaryPositionalEmbedding(dim=head_dim)
        q = torch.randn(2, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(2, n_heads, seq_len, head_dim, device=device)
        q_rot, k_rot = rope(q, k)
        assert q_rot.device == device
        
        # ALiBi
        alibi = ALiBiPositionalBias(n_heads=n_heads)
        bias = alibi(seq_len)
        assert bias.device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
