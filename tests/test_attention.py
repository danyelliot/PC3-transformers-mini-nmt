"""
Tests para componentes de atención y máscaras.
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.attention import (
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
    create_cross_attention_mask,
    combine_masks,
    KVCache
)


class TestMasks:
    """Tests para máscaras de atención."""
    
    def test_causal_mask_shape(self):
        """Verifica que la máscara causal tenga la forma correcta."""
        seq_len = 10
        mask = create_causal_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
    
    def test_causal_mask_values(self):
        """Verifica que la máscara causal sea triangular inferior."""
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        # Diagonal y debajo deben ser True
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == True, f"Position ({i},{j}) should be True"
                else:
                    assert mask[i, j] == False, f"Position ({i},{j}) should be False"
    
    def test_causal_mask_no_future_attention(self):
        """Verifica que token i no pueda atender a j > i."""
        seq_len = 4
        mask = create_causal_mask(seq_len)
        
        # Token en posición 0 solo se ve a sí mismo
        assert mask[0, 1:].sum() == 0
        
        # Token en posición 2 ve posiciones 0, 1, 2
        assert mask[2, :3].sum() == 3
        assert mask[2, 3] == False
    
    def test_padding_mask_shape(self):
        """Verifica shape de máscara de padding."""
        batch_size = 4
        seq_len = 10
        
        seq = torch.randint(1, 100, (batch_size, seq_len))
        mask = create_padding_mask(seq, pad_idx=0)
        
        assert mask.shape == (batch_size, 1, 1, seq_len)
    
    def test_padding_mask_values(self):
        """Verifica que padding tokens sean marcados correctamente."""
        seq = torch.tensor([
            [1, 2, 3, 0, 0],  # 3 tokens válidos
            [4, 5, 0, 0, 0],  # 2 tokens válidos
        ])
        
        mask = create_padding_mask(seq, pad_idx=0)
        
        # Remover dimensiones extra
        mask = mask.squeeze(1).squeeze(1)  # (batch, seq_len)
        
        # Verificar primera fila
        assert mask[0, 0] == True
        assert mask[0, 1] == True
        assert mask[0, 2] == True
        assert mask[0, 3] == False
        assert mask[0, 4] == False
        
        # Verificar segunda fila
        assert mask[1, 0] == True
        assert mask[1, 1] == True
        assert mask[1, 2] == False
    
    def test_combine_masks(self):
        """Verifica combinación de máscaras con AND lógico."""
        mask1 = torch.tensor([[True, True, False]])
        mask2 = torch.tensor([[True, False, False]])
        
        combined = combine_masks(mask1, mask2)
        
        expected = torch.tensor([[True, False, False]])
        assert torch.all(combined == expected)
    
    def test_cross_attention_mask(self):
        """Verifica máscara para cross-attention."""
        src_seq = torch.tensor([[1, 2, 3, 0, 0]])  # 3 válidos, 2 padding
        tgt_seq = torch.tensor([[4, 5, 6]])  # 3 válidos
        
        mask = create_cross_attention_mask(src_seq, tgt_seq, src_pad_idx=0)
        
        # Shape debe ser (batch, 1, tgt_len, src_len)
        assert mask.shape == (1, 1, 3, 5)
        
        # Primeros 3 tokens de src deben estar disponibles
        assert mask[0, 0, 0, 0] == True
        assert mask[0, 0, 0, 1] == True
        assert mask[0, 0, 0, 2] == True
        
        # Últimos 2 (padding) deben estar bloqueados
        assert mask[0, 0, 0, 3] == False
        assert mask[0, 0, 0, 4] == False


class TestScaledDotProductAttention:
    """Tests para Scaled Dot-Product Attention."""
    
    def test_attention_output_shape(self):
        """Verifica que el output tenga el shape correcto."""
        batch_size = 2
        n_heads = 4
        seq_len_q = 10
        seq_len_k = 8
        d_k = 32
        
        Q = torch.randn(batch_size, n_heads, seq_len_q, d_k)
        K = torch.randn(batch_size, n_heads, seq_len_k, d_k)
        V = torch.randn(batch_size, n_heads, seq_len_k, d_k)
        
        attn = ScaledDotProductAttention(dropout=0.0)
        output, _ = attn(Q, K, V)
        
        assert output.shape == (batch_size, n_heads, seq_len_q, d_k)
    
    def test_attention_with_causal_mask(self):
        """Verifica que la máscara causal funcione correctamente."""
        batch_size = 1
        n_heads = 1
        seq_len = 4
        d_k = 8
        
        Q = torch.randn(batch_size, n_heads, seq_len, d_k)
        K = torch.randn(batch_size, n_heads, seq_len, d_k)
        V = torch.randn(batch_size, n_heads, seq_len, d_k)
        
        # Crear máscara causal
        mask = create_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
        
        attn = ScaledDotProductAttention(dropout=0.0)
        output, attn_weights = attn(Q, K, V, mask=mask, return_attention=True)
        
        # Verificar que pesos de atención respeten la máscara
        # attn_weights[0, 0, i, j] debe ser ~0 si j > i
        attn_weights = attn_weights[0, 0]  # (seq_len, seq_len)
        
        # Token 0 no debe atender a tokens futuros
        assert attn_weights[0, 1] < 1e-6
        assert attn_weights[0, 2] < 1e-6
        assert attn_weights[0, 3] < 1e-6
        
        # Token 2 puede atender a 0, 1, 2 pero no a 3
        assert attn_weights[2, 3] < 1e-6
    
    def test_attention_weights_sum_to_one(self):
        """Verifica que los pesos de atención sumen 1 para cada query."""
        batch_size = 2
        n_heads = 2
        seq_len = 5
        d_k = 16
        
        Q = torch.randn(batch_size, n_heads, seq_len, d_k)
        K = torch.randn(batch_size, n_heads, seq_len, d_k)
        V = torch.randn(batch_size, n_heads, seq_len, d_k)
        
        attn = ScaledDotProductAttention(dropout=0.0)
        _, attn_weights = attn(Q, K, V, return_attention=True)
        
        # Suma sobre la última dimensión (seq_len_k) debe ser ~1
        sums = attn_weights.sum(dim=-1)
        
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestKVCache:
    """Tests para KV-cache."""
    
    def test_kv_cache_initialization(self):
        """Verifica inicialización del cache."""
        max_batch_size = 4
        max_seq_len = 128
        n_heads = 8
        head_dim = 32
        device = torch.device('cpu')
        
        cache = KVCache(max_batch_size, max_seq_len, n_heads, head_dim, device)
        
        assert cache.current_length == 0
        assert cache.cache_k.shape == (max_batch_size, n_heads, max_seq_len, head_dim)
    
    def test_kv_cache_update(self):
        """Verifica actualización del cache."""
        cache = KVCache(2, 64, 4, 16, torch.device('cpu'))
        
        # Primer update
        k1 = torch.randn(2, 4, 10, 16)
        v1 = torch.randn(2, 4, 10, 16)
        
        k_full, v_full = cache.update(k1, v1)
        
        assert cache.current_length == 10
        assert k_full.shape == (2, 4, 10, 16)
        
        # Segundo update
        k2 = torch.randn(2, 4, 5, 16)
        v2 = torch.randn(2, 4, 5, 16)
        
        k_full, v_full = cache.update(k2, v2)
        
        assert cache.current_length == 15
        assert k_full.shape == (2, 4, 15, 16)
    
    def test_kv_cache_reset(self):
        """Verifica reset del cache."""
        cache = KVCache(2, 64, 4, 16, torch.device('cpu'))
        
        k = torch.randn(2, 4, 10, 16)
        v = torch.randn(2, 4, 10, 16)
        cache.update(k, v)
        
        assert cache.current_length == 10
        
        cache.reset()
        
        assert cache.current_length == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
