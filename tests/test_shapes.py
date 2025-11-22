"""
Tests para shapes y dimensiones de los componentes.
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import (
    Transformer,
    TransformerConfig,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer
)
from src.models.mhsa import (
    MultiHeadAttention,
    MultiHeadSelfAttention,
    MultiHeadCrossAttention,
    FeedForwardNetwork
)


class TestMultiHeadAttention:
    """Tests para Multi-Head Attention."""
    
    def test_mhsa_output_shape(self):
        """Verifica shape de output de MHSA."""
        batch_size = 4
        seq_len = 32
        d_model = 256
        n_heads = 8
        
        mhsa = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, _ = mhsa(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_mhsa_attention_weights_shape(self):
        """Verifica shape de los pesos de atención."""
        batch_size = 2
        seq_len = 16
        d_model = 128
        n_heads = 4
        
        mhsa = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = mhsa(x, return_attention=True)
        
        # attn_weights: (batch, n_heads, seq_len, seq_len)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_cross_attention_shapes(self):
        """Verifica shapes en cross-attention."""
        batch_size = 4
        src_len = 20
        tgt_len = 15
        d_model = 256
        n_heads = 8
        
        cross_attn = MultiHeadCrossAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
        
        query = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        
        output, attn_weights = cross_attn(query, encoder_output, return_attention=True)
        
        assert output.shape == (batch_size, tgt_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, tgt_len, src_len)
    
    def test_ffn_shape(self):
        """Verifica shape de Feed-Forward Network."""
        batch_size = 4
        seq_len = 32
        d_model = 256
        dim_feedforward = 1024
        
        ffn = FeedForwardNetwork(d_model=d_model, dim_feedforward=dim_feedforward, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ffn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerLayers:
    """Tests para capas del Transformer."""
    
    def test_encoder_layer_shape(self):
        """Verifica shape de EncoderLayer."""
        batch_size = 4
        seq_len = 32
        d_model = 256
        n_heads = 8
        dim_feedforward = 1024
        
        encoder_layer = EncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = encoder_layer(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_decoder_layer_shape(self):
        """Verifica shape de DecoderLayer."""
        batch_size = 4
        src_len = 20
        tgt_len = 15
        d_model = 256
        n_heads = 8
        dim_feedforward = 1024
        
        decoder_layer = DecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0
        )
        
        tgt = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        
        output = decoder_layer(tgt, encoder_output)
        
        assert output.shape == (batch_size, tgt_len, d_model)


class TestTransformerModel:
    """Tests para el modelo Transformer completo."""
    
    def test_transformer_forward_shape(self):
        """Verifica shape del forward completo."""
        config = TransformerConfig(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            vocab_size_src=10000,
            vocab_size_tgt=10000,
            max_seq_length=128,
            dropout=0.0
        )
        
        model = Transformer(config)
        
        batch_size = 4
        src_len = 20
        tgt_len = 15
        
        src = torch.randint(0, 10000, (batch_size, src_len))
        tgt = torch.randint(0, 10000, (batch_size, tgt_len))
        
        logits = model(src, tgt)
        
        # logits: (batch, tgt_len, vocab_size_tgt)
        assert logits.shape == (batch_size, tgt_len, config.vocab_size_tgt)
    
    def test_encoder_output_shape(self):
        """Verifica shape del encoder."""
        config = TransformerConfig(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            vocab_size_src=10000,
            max_seq_length=128,
            dropout=0.0
        )
        
        model = Transformer(config)
        
        batch_size = 4
        src_len = 20
        src = torch.randint(0, 10000, (batch_size, src_len))
        
        encoder_output = model.encode(src)
        
        # (batch, src_len, d_model)
        assert encoder_output.shape == (batch_size, src_len, config.d_model)
    
    def test_decoder_step_shape(self):
        """Verifica shape del decode_step."""
        config = TransformerConfig(
            d_model=256,
            nhead=8,
            num_decoder_layers=4,
            vocab_size_tgt=10000,
            max_seq_length=128,
            dropout=0.0
        )
        
        model = Transformer(config)
        
        batch_size = 4
        src_len = 20
        tgt_len = 10
        
        src = torch.randint(0, 10000, (batch_size, src_len))
        tgt = torch.randint(0, 10000, (batch_size, tgt_len))
        
        encoder_output = model.encode(src)
        logits = model.decode_step(tgt, encoder_output)
        
        assert logits.shape == (batch_size, tgt_len, config.vocab_size_tgt)
    
    def test_different_src_tgt_lengths(self):
        """Verifica que funcione con diferentes longitudes src/tgt."""
        config = TransformerConfig(
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            vocab_size_src=5000,
            vocab_size_tgt=5000,
            dropout=0.0
        )
        
        model = Transformer(config)
        
        # Longitudes muy diferentes
        src = torch.randint(0, 5000, (2, 50))
        tgt = torch.randint(0, 5000, (2, 10))
        
        logits = model(src, tgt)
        
        assert logits.shape == (2, 10, 5000)
    
    def test_batch_size_one(self):
        """Verifica que funcione con batch_size=1."""
        config = TransformerConfig(
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            vocab_size_src=1000,
            vocab_size_tgt=1000,
            dropout=0.0
        )
        
        model = Transformer(config)
        
        src = torch.randint(0, 1000, (1, 20))
        tgt = torch.randint(0, 1000, (1, 15))
        
        logits = model(src, tgt)
        
        assert logits.shape == (1, 15, 1000)


class TestParameterCounts:
    """Tests para contar parámetros."""
    
    def test_model_has_parameters(self):
        """Verifica que el modelo tenga parámetros."""
        config = TransformerConfig(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            vocab_size_src=10000,
            vocab_size_tgt=10000
        )
        
        model = Transformer(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # Todos deben ser entrenables
    
    def test_parameter_count_increases_with_layers(self):
        """Verifica que más capas = más parámetros."""
        config_2_layers = TransformerConfig(
            d_model=256,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            vocab_size_src=5000,
            vocab_size_tgt=5000
        )
        
        config_4_layers = TransformerConfig(
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            vocab_size_src=5000,
            vocab_size_tgt=5000
        )
        
        model_2 = Transformer(config_2_layers)
        model_4 = Transformer(config_4_layers)
        
        params_2 = sum(p.numel() for p in model_2.parameters())
        params_4 = sum(p.numel() for p in model_4.parameters())
        
        assert params_4 > params_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
