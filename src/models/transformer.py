"""
Transformer completo para NMT (Neural Machine Translation).

Implementa:
- Encoder bidireccional (atención sin máscara causal)
- Decoder autoregresivo (con self-attention causal y cross-attention)
- Embeddings compartidos opcionales
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from .mhsa import MultiHeadSelfAttention, MultiHeadCrossAttention, FeedForwardNetwork
from .posenc import SinusoidalPositionalEncoding, RotaryPositionalEmbedding, ALiBiPositionalBias
from .attention import create_causal_mask, create_padding_mask, combine_masks


@dataclass
class TransformerConfig:
    """Configuración del Transformer."""
    # Dimensiones
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Vocabularios
    vocab_size_src: int = 10000
    vocab_size_tgt: int = 10000
    max_seq_length: int = 128
    
    # Codificación posicional
    pos_encoding: str = "sinusoidal"  # sinusoidal, rope, alibi
    
    # Otros
    share_embeddings: bool = False  # Compartir embeddings src-tgt
    pad_idx: int = 0


class TransformerEmbedding(nn.Module):
    """
    Embedding + Positional Encoding.
    
    Args:
        vocab_size: Tamaño del vocabulario
        d_model: Dimensión del modelo
        max_len: Longitud máxima
        dropout: Tasa de dropout
        pos_encoding_type: Tipo de codificación posicional
        pad_idx: Índice del token de padding
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
        pos_encoding_type: str = "sinusoidal",
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        if pos_encoding_type == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding_type == "rope":
            # RoPE se aplica directamente en la atención, no aquí
            self.pos_encoding = None
            self.dropout = nn.Dropout(dropout)
        elif pos_encoding_type == "alibi":
            # ALiBi se aplica como sesgo en la atención
            self.pos_encoding = None
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"pos_encoding_type '{pos_encoding_type}' no soportado")
        
        # Escalado según el paper original
        self.scale = math.sqrt(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Índices de tokens (batch, seq_len)
            
        Returns:
            Embeddings (batch, seq_len, d_model)
        """
        # Token embedding escalado
        embeddings = self.token_embedding(x) * self.scale
        
        # Añadir positional encoding si es sinusoidal
        if self.pos_encoding_type == "sinusoidal":
            embeddings = self.pos_encoding(embeddings)
        else:
            embeddings = self.dropout(embeddings)
        
        return embeddings


class EncoderLayer(nn.Module):
    """
    Una capa del Encoder del Transformer.
    
    Arquitectura:
    1. Multi-Head Self-Attention (bidireccional)
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        rotary=None,
        alibi=None
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        # inyectar pos-encodings si son provistos
        if rotary is not None:
            self.self_attn.rotary = rotary
        if alibi is not None:
            self.self_attn.alibi = alibi
        
        # Feed-forward
        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        
        # Layer normalization (pre-norm como en GPT)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Máscara de padding (batch, 1, 1, seq_len)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm + Self-attention + Residual
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, mask=mask)
        x = residual + x
        
        # Pre-norm + FFN + Residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class DecoderLayer(nn.Module):
    """
    Una capa del Decoder del Transformer.
    
    Arquitectura:
    1. Masked Multi-Head Self-Attention (causal)
    2. Add & Norm
    3. Multi-Head Cross-Attention (al encoder)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        rotary=None,
        alibi=None
    ):
        super().__init__()
        
        # Masked self-attention (causal)
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        # Cross-attention al encoder
        self.cross_attn = MultiHeadCrossAttention(d_model, nhead, dropout)
        # inyectar pos-encodings si son provistos
        if rotary is not None:
            self.self_attn.rotary = rotary
            self.cross_attn.rotary = rotary
        if alibi is not None:
            self.self_attn.alibi = alibi
            self.cross_attn.alibi = alibi
        
        # Feed-forward
        self.ffn = FeedForwardNetwork(d_model, dim_feedforward, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Entrada del decoder (batch, tgt_len, d_model)
            encoder_output: Salida del encoder (batch, src_len, d_model)
            tgt_mask: Máscara causal + padding para target (batch, 1, tgt_len, tgt_len)
            memory_mask: Máscara de padding para source (batch, 1, tgt_len, src_len)
            
        Returns:
            output: (batch, tgt_len, d_model)
        """
        # Pre-norm + Masked Self-Attention + Residual
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, mask=tgt_mask)
        x = residual + x

        # Pre-norm + Cross-Attention + Residual
        residual = x
        x = self.norm2(x)
        if return_attention:
            x, cross_attn = self.cross_attn(x, encoder_output, mask=memory_mask, return_attention=True)
        else:
            x, _ = self.cross_attn(x, encoder_output, mask=memory_mask)
            cross_attn = None
        x = residual + x

        # Pre-norm + FFN + Residual
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x

        if return_attention:
            return x, cross_attn
        return x


class Encoder(nn.Module):
    """
    Stack de capas del Encoder.
    """
    
    def __init__(self, config: TransformerConfig, rotary=None, alibi=None):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                rotary=rotary,
                alibi=alibi
            )
            for _ in range(config.num_encoder_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Máscara de padding (batch, 1, 1, seq_len)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class Decoder(nn.Module):
    """
    Stack de capas del Decoder.
    """
    
    def __init__(self, config: TransformerConfig, rotary=None, alibi=None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                rotary=rotary,
                alibi=alibi
            )
            for _ in range(config.num_decoder_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model)
            encoder_output: (batch, src_len, d_model)
            tgt_mask: Máscara para target (batch, 1, tgt_len, tgt_len)
            memory_mask: Máscara para source (batch, 1, tgt_len, src_len)
            
        Returns:
            output: (batch, tgt_len, d_model)
        """
        # Support optionally returning cross-attention weights from each layer
        attn_list = []
        for layer in self.layers:
            if isinstance(layer, DecoderLayer):
                # propagate the return_attention flag to each layer
                if return_attention:
                    x, attn = layer(x, encoder_output, tgt_mask, memory_mask, return_attention=True)
                    attn_list.append(attn)
                else:
                    x = layer(x, encoder_output, tgt_mask, memory_mask, return_attention=False)
            else:
                # Fallback: call layer without attention
                x = layer(x, encoder_output, tgt_mask, memory_mask)

        x = self.norm(x)
        if return_attention:
            return x, attn_list
        return x


class Transformer(nn.Module):
    """
    Transformer completo para NMT (seq2seq).
    
    Arquitectura:
    - Encoder: embeddings -> N capas encoder -> output
    - Decoder: embeddings -> N capas decoder (con cross-attention) -> linear -> logits
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.src_embedding = TransformerEmbedding(
            vocab_size=config.vocab_size_src,
            d_model=config.d_model,
            max_len=config.max_seq_length,
            dropout=config.dropout,
            pos_encoding_type=config.pos_encoding,
            pad_idx=config.pad_idx
        )
        
        if config.share_embeddings and config.vocab_size_src == config.vocab_size_tgt:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = TransformerEmbedding(
                vocab_size=config.vocab_size_tgt,
                d_model=config.d_model,
                max_len=config.max_seq_length,
                dropout=config.dropout,
                pos_encoding_type=config.pos_encoding,
                pad_idx=config.pad_idx
            )
        
        # Preparar pos-encodings para inyectar en las capas
        if config.pos_encoding == 'rope':
            head_dim = config.d_model // config.nhead
            self.rotary = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=config.max_seq_length)
        else:
            self.rotary = None

        # ALiBi si es necesario (crear antes de construir capas)
        if config.pos_encoding == "alibi":
            self.alibi = ALiBiPositionalBias(config.nhead, config.max_seq_length)
        else:
            self.alibi = None

        # Encoder y Decoder (inyectar rotary/alibi si aplica)
        self.encoder = Encoder(config, rotary=self.rotary, alibi=self.alibi)
        self.decoder = Decoder(config, rotary=self.rotary, alibi=self.alibi)
        
        # Proyección final a vocabulario
        self.output_projection = nn.Linear(config.d_model, config.vocab_size_tgt)
        
        # ALiBi si es necesario
        if config.pos_encoding == "alibi":
            self.alibi = ALiBiPositionalBias(config.nhead, config.max_seq_length)
        else:
            self.alibi = None
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos del modelo."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass del Transformer.
        
        Args:
            src: Índices source (batch, src_len)
            tgt: Índices target (batch, tgt_len)
            src_mask: Máscara source (opcional, se crea automáticamente)
            tgt_mask: Máscara target causal (opcional, se crea automáticamente)
            memory_mask: Máscara cross-attention (opcional, se crea automáticamente)
            
        Returns:
            logits: (batch, tgt_len, vocab_size_tgt)
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        device = src.device
        
        # Crear máscaras si no se proporcionan
        if src_mask is None:
            src_mask = create_padding_mask(src, self.config.pad_idx)
        
        if tgt_mask is None:
            # Combinar máscara causal y de padding
            causal_mask = create_causal_mask(tgt_len, device)  # (tgt_len, tgt_len)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
            
            padding_mask = create_padding_mask(tgt, self.config.pad_idx)  # (batch, 1, 1, tgt_len)
            padding_mask = padding_mask.expand(-1, -1, tgt_len, -1)  # (batch, 1, tgt_len, tgt_len)
            
            tgt_mask = causal_mask & padding_mask
        
        if memory_mask is None:
            # Máscara para cross-attention (solo padding del source)
            memory_mask = create_padding_mask(src, self.config.pad_idx)  # (batch, 1, 1, src_len)
            memory_mask = memory_mask.expand(-1, -1, tgt_len, -1)  # (batch, 1, tgt_len, src_len)
        
        # Embeddings
        src_emb = self.src_embedding(src)  # (batch, src_len, d_model)
        tgt_emb = self.tgt_embedding(tgt)  # (batch, tgt_len, d_model)
        
        # Encoder
        encoder_output = self.encoder(src_emb, src_mask)  # (batch, src_len, d_model)
        
        # Decoder
        decoder_output = self.decoder(
            tgt_emb,
            encoder_output,
            tgt_mask,
            memory_mask
        )  # (batch, tgt_len, d_model)
        
        # Proyección a vocabulario
        logits = self.output_projection(decoder_output)  # (batch, tgt_len, vocab_size_tgt)
        
        return logits
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Solo la parte del encoder (útil para generación).
        
        Args:
            src: (batch, src_len)
            src_mask: Máscara opcional
            
        Returns:
            encoder_output: (batch, src_len, d_model)
        """
        if src_mask is None:
            src_mask = create_padding_mask(src, self.config.pad_idx)
        
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb, src_mask)
    
    def decode_step(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Un paso de decodificación (útil para generación autoregresiva).
        
        Args:
            tgt: (batch, tgt_len)
            encoder_output: (batch, src_len, d_model)
            tgt_mask: Máscara opcional
            memory_mask: Máscara opcional
            
        Returns:
            logits: (batch, tgt_len, vocab_size_tgt)
        """
        tgt_len = tgt.size(1)
        device = tgt.device
        
        # default causal mask if not provided
        if tgt_mask is None:
            causal_mask = create_causal_mask(tgt_len, device)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
            tgt_mask = causal_mask

        tgt_emb = self.tgt_embedding(tgt)
        # Allow optionally returning attention weights from cross-attention
        # Keep backward compatibility: default return only logits
        # If caller passed return_attention=True, forward that flag to decoder
        decoder_result = self.decoder(tgt_emb, encoder_output, tgt_mask, memory_mask, return_attention=return_attention)
        if isinstance(decoder_result, tuple):
            decoder_output, attn_list = decoder_result
            logits = self.output_projection(decoder_output)
            # Aggregate attentions across layers by averaging (skip None)
            valid_attns = [a for a in attn_list if a is not None]
            if valid_attns:
                # stack -> (n_layers, batch, n_heads, tgt_len, src_len)
                attn_stack = torch.stack(valid_attns, dim=0)
                # mean over layers -> (batch, n_heads, tgt_len, src_len)
                attn_agg = attn_stack.mean(dim=0)
            else:
                attn_agg = None

            return logits, attn_agg
        else:
            logits = self.output_projection(decoder_result)
            return logits


def create_transformer_from_config(config_dict: dict) -> Transformer:
    """
    Crea un Transformer desde un diccionario de configuración.
    
    Args:
        config_dict: Diccionario con parámetros
        
    Returns:
        Modelo Transformer
    """
    config = TransformerConfig(**config_dict.get('model', {}))
    return Transformer(config)
