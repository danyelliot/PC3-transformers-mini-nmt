#!/usr/bin/env python3
"""
Ejemplo mínimo para probar encode + greedy decode con el modelo Transformer del repo.
Ejecutar:
    python examples/run_demo.py
"""
import torch

from src.models.transformer import Transformer, TransformerConfig
from src.decoding import Decoder, DecodingConfig


def main():
    cfg = TransformerConfig(
        d_model=128, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2,
        vocab_size_src=1000, vocab_size_tgt=1000,
        max_seq_length=64, dropout=0.0
    )
    model = Transformer(cfg)
    model.eval()

    decoding_cfg = DecodingConfig(strategy="greedy", max_length=20)
    decoder = Decoder(model, decoding_cfg, bos_token_id=1, eos_token_id=2, pad_token_id=0, device=torch.device('cpu'))

    # batch de ejemplo (valores aleatorios en vocab)
    src = torch.randint(3, 1000, (2, 10))
    print("src shape:", src.shape)

    out = decoder.greedy_search(src)
    # decoder.greedy_search returns a Python list of sequences
    if isinstance(out, list):
        print("Greedy output: list with batch size", len(out))
        for i, seq in enumerate(out):
            print(f"  Example {i}: length={len(seq)} ->", seq)
    else:
        print("Greedy output shape:", out.shape)
        print(out)


if __name__ == "__main__":
    main()
