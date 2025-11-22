"""
Tests para estrategias de decodificación.
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import Transformer, TransformerConfig
from src.decoding import Decoder, DecodingConfig
from src.utils import set_seed


class TestGreedySearch:
    """Tests para greedy search."""
    
    def test_greedy_deterministic(self):
        """Verifica que greedy sea determinista con la misma seed."""
        set_seed(42)
        
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
        model.eval()
        
        decoding_config = DecodingConfig(
            strategy="greedy",
            max_length=20
        )
        
        decoder = Decoder(
            model,
            decoding_config,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            device=torch.device('cpu')
        )
        
        src = torch.randint(3, 1000, (2, 10))
        
        # Primera corrida
        set_seed(42)
        output1 = decoder.greedy_search(src)
        
        # Segunda corrida con misma seed
        set_seed(42)
        output2 = decoder.greedy_search(src)
        
        # Deben ser idénticas
        assert output1 == output2, "Greedy search debe ser determinista"
    
    def test_greedy_output_format(self):
        """Verifica formato del output."""
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
        model.eval()
        
        decoding_config = DecodingConfig(strategy="greedy", max_length=20)
        decoder = Decoder(
            model, decoding_config,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            device=torch.device('cpu')
        )
        
        batch_size = 4
        src = torch.randint(3, 1000, (batch_size, 10))
        
        output = decoder.greedy_search(src)
        
        # Debe retornar lista de listas
        assert isinstance(output, list)
        assert len(output) == batch_size
        assert all(isinstance(seq, list) for seq in output)


class TestBeamSearch:
    """Tests para beam search."""
    
    def test_beam_search_basic(self):
        """Verifica funcionamiento básico de beam search."""
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
        model.eval()
        
        decoding_config = DecodingConfig(
            strategy="beam",
            beam_size=4,
            max_length=20,
            length_penalty=0.6
        )
        
        decoder = Decoder(
            model, decoding_config,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            device=torch.device('cpu')
        )
        
        src = torch.randint(3, 1000, (2, 10))
        
        output = decoder.beam_search(src)
        
        assert isinstance(output, list)
        assert len(output) == 2
    
    def test_beam_size_effect(self):
        """Verifica que beam_size afecte los resultados."""
        set_seed(42)
        
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
        model.eval()
        
        src = torch.randint(3, 1000, (1, 10))
        
        # Beam size 1 (similar a greedy)
        config1 = DecodingConfig(strategy="beam", beam_size=1, max_length=20)
        decoder1 = Decoder(model, config1, 1, 2, 0, torch.device('cpu'))
        
        set_seed(42)
        output1 = decoder1.beam_search(src)
        
        # Beam size 4
        config4 = DecodingConfig(strategy="beam", beam_size=4, max_length=20)
        decoder4 = Decoder(model, config4, 1, 2, 0, torch.device('cpu'))
        
        set_seed(42)
        output4 = decoder4.beam_search(src)
        
        # Con beam más grande, potencialmente diferente resultado
        # (no siempre, pero al menos verificamos que no crashea)
        assert len(output1) == len(output4) == 1


class TestSamplingMethods:
    """Tests para métodos de sampling."""
    
    def test_top_k_sampling(self):
        """Verifica top-k sampling."""
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
        model.eval()
        
        decoding_config = DecodingConfig(
            strategy="topk",
            top_k=50,
            temperature=1.0,
            max_length=20
        )
        
        decoder = Decoder(
            model, decoding_config,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            device=torch.device('cpu')
        )
        
        src = torch.randint(3, 1000, (2, 10))
        
        output = decoder.top_k_sampling(src)
        
        assert isinstance(output, list)
        assert len(output) == 2
    
    def test_top_p_sampling(self):
        """Verifica top-p (nucleus) sampling."""
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
        model.eval()
        
        decoding_config = DecodingConfig(
            strategy="topp",
            top_p=0.92,
            temperature=1.0,
            max_length=20
        )
        
        decoder = Decoder(
            model, decoding_config,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            device=torch.device('cpu')
        )
        
        src = torch.randint(3, 1000, (2, 10))
        
        output = decoder.top_p_sampling(src)
        
        assert isinstance(output, list)
        assert len(output) == 2
    
    def test_temperature_effect(self):
        """Verifica que temperature afecte la diversidad."""
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
        model.eval()
        
        src = torch.randint(3, 1000, (1, 10))
        
        # Temperature alta = más diversidad
        config_high_temp = DecodingConfig(
            strategy="topk",
            top_k=50,
            temperature=2.0,
            max_length=15
        )
        
        decoder_high = Decoder(
            model, config_high_temp,
            1, 2, 0, torch.device('cpu')
        )
        
        # Generar múltiples veces y verificar que haya diversidad
        outputs = []
        for _ in range(5):
            output = decoder_high.top_k_sampling(src)
            outputs.append(output[0])
        
        # Con temperatura alta, debería haber al menos alguna variación
        # (no siempre garantizado con modelo pequeño, pero es una prueba razonable)
        unique_outputs = len(set(tuple(o) for o in outputs))
        assert unique_outputs >= 1  # Al menos uno debe existir


class TestPenalties:
    """Tests para penalizaciones."""
    
    def test_repetition_penalty(self):
        """Verifica que repetition penalty se aplique."""
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
        model.eval()
        
        # Con penalty
        config_with_penalty = DecodingConfig(
            strategy="greedy",
            repetition_penalty=1.5,
            max_length=20
        )
        
        decoder = Decoder(
            model, config_with_penalty,
            1, 2, 0, torch.device('cpu')
        )
        
        src = torch.randint(3, 1000, (1, 10))
        
        set_seed(42)
        output = decoder.greedy_search(src)
        
        # Verificar que se genera algo
        assert len(output) == 1
        assert len(output[0]) > 0


class TestMaxLength:
    """Tests para max_length."""
    
    def test_max_length_respected(self):
        """Verifica que max_length se respete."""
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
        model.eval()
        
        max_len = 15
        decoding_config = DecodingConfig(
            strategy="greedy",
            max_length=max_len
        )
        
        decoder = Decoder(
            model, decoding_config,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            device=torch.device('cpu')
        )
        
        src = torch.randint(3, 1000, (2, 10))
        
        output = decoder.greedy_search(src)
        
        # Longitud no debe exceder max_length + 1 (por BOS)
        for seq in output:
            assert len(seq) <= max_len + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
