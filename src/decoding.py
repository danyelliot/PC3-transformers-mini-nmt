"""
Estrategias de decodificación para generación de secuencias.

Implementa múltiples métodos de decodificación:
- Greedy search
- Beam search con length penalty y coverage penalty
- Top-k sampling
- Top-p (nucleus) sampling
- Penalizaciones por repetición y frecuencia
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math
from dataclasses import dataclass


@dataclass
class DecodingConfig:
    """Configuración para decodificación."""
    strategy: str = "beam"  # greedy, beam, topk, topp
    beam_size: int = 4
    length_penalty: float = 0.6
    coverage_penalty: float = 0.2
    repetition_penalty: float = 1.2
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.92
    max_length: int = 128
    min_length: int = 1
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True


class Decoder:
    """
    Clase para decodificación de secuencias con múltiples estrategias.
    """
    
    def __init__(
        self,
        model,
        config: DecodingConfig,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        device: torch.device
    ):
        """
        Args:
            model: Modelo Transformer
            config: Configuración de decodificación
            bos_token_id: ID del token de inicio de secuencia
            eos_token_id: ID del token de fin de secuencia
            pad_token_id: ID del token de padding
            device: Dispositivo (CPU/CUDA)
        """
        self.model = model
        self.config = config
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.device = device
    
    def decode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Decodifica usando la estrategia configurada.
        
        Args:
            src: Tensor de entrada (batch, src_len)
            src_mask: Máscara opcional
            
        Returns:
            Lista de secuencias decodificadas (batch_size, seq_len)
        """
        if self.config.strategy == "greedy":
            return self.greedy_search(src, src_mask)
        elif self.config.strategy == "beam":
            return self.beam_search(src, src_mask)
        elif self.config.strategy == "topk":
            return self.top_k_sampling(src, src_mask)
        elif self.config.strategy == "topp":
            return self.top_p_sampling(src, src_mask)
        else:
            raise ValueError(f"Estrategia '{self.config.strategy}' no soportada")
    
    def greedy_search(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Greedy search: en cada paso selecciona el token con mayor probabilidad.
        
        Args:
            src: (batch, src_len)
            src_mask: Máscara opcional
            
        Returns:
            Lista de secuencias (batch_size, variable_len)
        """
        batch_size = src.size(0)
        
        # Codificar source
        encoder_output = self.model.encode(src, src_mask)
        
        # Inicializar con BOS
        tgt = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        
        # Track de secuencias terminadas
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.config.max_length):
            # Decodificar un paso
            logits = self.model.decode_step(tgt, encoder_output, memory_mask=src_mask)
            
            # Tomar logits del último token generado
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Aplicar penalizaciones
            next_token_logits = self._apply_penalties(next_token_logits, tgt)
            
            # Seleccionar token con máxima probabilidad
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch, 1)
            
            # Actualizar secuencias no terminadas
            next_tokens = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_tokens, self.pad_token_id),
                next_tokens
            )
            
            # Concatenar
            tgt = torch.cat([tgt, next_tokens], dim=1)
            
            # Actualizar secuencias terminadas
            finished = finished | (next_tokens.squeeze(1) == self.eos_token_id)
            
            # Si todas terminaron, parar
            if finished.all():
                break
        
        # Convertir a lista
        return tgt.tolist()
    
    def beam_search(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Beam search con length penalty y coverage penalty.
        
        Mantiene beam_size hipótesis y selecciona la mejor secuencia global.
        
        Args:
            src: (batch, src_len)
            src_mask: Máscara opcional
            
        Returns:
            Lista de mejores secuencias para cada ejemplo del batch
        """
        batch_size = src.size(0)
        beam_size = self.config.beam_size
        
        # Codificar source
        encoder_output = self.model.encode(src, src_mask)
        
        # Expandir encoder_output para beam search
        # (batch, src_len, d_model) -> (batch * beam_size, src_len, d_model)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, beam_size, 1, 1)
        encoder_output = encoder_output.view(batch_size * beam_size, -1, encoder_output.size(-1))
        
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
            src_mask = src_mask.view(batch_size * beam_size, *src_mask.shape[2:])
        
        # Inicializar beams
        beams = torch.full(
            (batch_size * beam_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Scores de cada beam (log probabilidades acumuladas)
        beam_scores = torch.zeros(batch_size, beam_size, device=self.device)
        beam_scores[:, 1:] = float('-inf')  # Solo el primer beam está activo inicialmente
        beam_scores = beam_scores.view(-1)  # (batch * beam_size,)
        
        # Track de beams terminados
        finished_beams = [[] for _ in range(batch_size)]
        
        for step in range(self.config.max_length):
            # Decodificar
            logits = self.model.decode_step(beams, encoder_output, memory_mask=src_mask)
            next_token_logits = logits[:, -1, :]  # (batch * beam_size, vocab_size)
            
            # Aplicar temperature
            if self.config.temperature != 1.0:
                next_token_logits = next_token_logits / self.config.temperature
            
            # Log probabilities
            log_probs = F.log_softmax(next_token_logits, dim=-1)  # (batch * beam_size, vocab_size)
            
            # Aplicar penalizaciones
            log_probs = self._apply_penalties_beam(log_probs, beams)
            
            vocab_size = log_probs.size(-1)
            
            # Scores para cada posible próximo token
            # (batch * beam_size, vocab_size) + (batch * beam_size, 1)
            next_scores = log_probs + beam_scores.unsqueeze(1)
            
            # Reshape para seleccionar top-k entre todos los beams
            next_scores = next_scores.view(batch_size, beam_size * vocab_size)
            
            # Seleccionar top-2*beam_size para cada batch
            # (para poder filtrar EOS y mantener beam_size activos)
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            
            # Calcular de qué beam viene cada token y cuál es el token
            next_beam_indices = next_tokens // vocab_size  # (batch, 2*beam_size)
            next_tokens = next_tokens % vocab_size  # (batch, 2*beam_size)
            
            # Para cada ejemplo del batch, actualizar beams
            new_beams = []
            new_scores = []
            
            for batch_idx in range(batch_size):
                beam_idx_for_batch = batch_idx * beam_size
                active_beams = []
                
                for candidate_idx in range(2 * beam_size):
                    token = next_tokens[batch_idx, candidate_idx].item()
                    score = next_scores[batch_idx, candidate_idx].item()
                    beam_idx = next_beam_indices[batch_idx, candidate_idx].item()
                    
                    # Reconstruir secuencia
                    prev_beam_seq = beams[beam_idx_for_batch + beam_idx]
                    new_seq = torch.cat([prev_beam_seq, torch.tensor([token], device=self.device)])
                    
                    # Si es EOS, guardar en finished_beams
                    if token == self.eos_token_id:
                        # Aplicar length penalty
                        length = len(new_seq)
                        normalized_score = score / (length ** self.config.length_penalty)
                        finished_beams[batch_idx].append((normalized_score, new_seq))
                    else:
                        active_beams.append((score, new_seq, beam_idx))
                    
                    # Si ya tenemos beam_size beams activos, parar
                    if len(active_beams) >= beam_size:
                        break
                
                # Si no hay suficientes beams activos, rellenar con los mejores finished
                if len(active_beams) < beam_size and finished_beams[batch_idx]:
                    finished_beams[batch_idx].sort(reverse=True, key=lambda x: x[0])
                    for score, seq in finished_beams[batch_idx][:beam_size - len(active_beams)]:
                        active_beams.append((score, seq, 0))
                
                # Añadir beams activos
                for score, seq, _ in active_beams[:beam_size]:
                    new_beams.append(seq)
                    new_scores.append(score)
            
            # Actualizar beams
            if len(new_beams) == 0:
                break
            
            # Pad secuencias al mismo tamaño
            max_len = max(len(seq) for seq in new_beams)
            beams = torch.stack([
                F.pad(seq, (0, max_len - len(seq)), value=self.pad_token_id)
                for seq in new_beams
            ])
            
            beam_scores = torch.tensor(new_scores, device=self.device)
            
            # Early stopping si todos los batches tienen beams terminados
            if self.config.early_stopping and all(len(fb) >= beam_size for fb in finished_beams):
                break
        
        # Seleccionar mejores secuencias
        results = []
        for batch_idx in range(batch_size):
            if finished_beams[batch_idx]:
                # Ordenar por score normalizado
                finished_beams[batch_idx].sort(reverse=True, key=lambda x: x[0])
                best_seq = finished_beams[batch_idx][0][1]
            else:
                # Si no hay beams terminados, tomar el primero
                best_seq = beams[batch_idx * beam_size]
            
            results.append(best_seq.tolist())
        
        return results
    
    def top_k_sampling(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Top-k sampling: muestrea solo entre los k tokens más probables.
        
        Args:
            src: (batch, src_len)
            src_mask: Máscara opcional
            
        Returns:
            Lista de secuencias
        """
        batch_size = src.size(0)
        encoder_output = self.model.encode(src, src_mask)
        
        tgt = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.config.max_length):
            logits = self.model.decode_step(tgt, encoder_output, memory_mask=src_mask)
            next_token_logits = logits[:, -1, :] / self.config.temperature
            
            # Aplicar penalizaciones
            next_token_logits = self._apply_penalties(next_token_logits, tgt)
            
            # Top-k filtering
            top_k = min(self.config.top_k, next_token_logits.size(-1))
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            next_tokens = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_tokens, self.pad_token_id),
                next_tokens
            )
            
            tgt = torch.cat([tgt, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(1) == self.eos_token_id)
            
            if finished.all():
                break
        
        return tgt.tolist()
    
    def top_p_sampling(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Top-p (nucleus) sampling: muestrea del conjunto mínimo que suma probabilidad >= p.
        
        Args:
            src: (batch, src_len)
            src_mask: Máscara opcional
            
        Returns:
            Lista de secuencias
        """
        batch_size = src.size(0)
        encoder_output = self.model.encode(src, src_mask)
        
        tgt = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.config.max_length):
            logits = self.model.decode_step(tgt, encoder_output, memory_mask=src_mask)
            next_token_logits = logits[:, -1, :] / self.config.temperature
            
            # Aplicar penalizaciones
            next_token_logits = self._apply_penalties(next_token_logits, tgt)
            
            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remover tokens con probabilidad acumulada > p
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            # Mantener al menos el primer token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for batch_idx in range(batch_size):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            next_tokens = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_tokens, self.pad_token_id),
                next_tokens
            )
            
            tgt = torch.cat([tgt, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(1) == self.eos_token_id)
            
            if finished.all():
                break
        
        return tgt.tolist()
    
    def _apply_penalties(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica penalizaciones por repetición y frecuencia.
        
        Args:
            logits: (batch, vocab_size)
            generated_tokens: (batch, seq_len)
            
        Returns:
            logits modificados
        """
        batch_size, vocab_size = logits.shape
        
        # Repetition penalty
        if self.config.repetition_penalty != 1.0:
            for batch_idx in range(batch_size):
                for token in set(generated_tokens[batch_idx].tolist()):
                    if token == self.pad_token_id:
                        continue
                    # Si el logit es positivo, dividir; si negativo, multiplicar
                    if logits[batch_idx, token] < 0:
                        logits[batch_idx, token] *= self.config.repetition_penalty
                    else:
                        logits[batch_idx, token] /= self.config.repetition_penalty
        
        # Frequency penalty
        if self.config.frequency_penalty != 0.0:
            for batch_idx in range(batch_size):
                token_counts = {}
                for token in generated_tokens[batch_idx].tolist():
                    if token == self.pad_token_id:
                        continue
                    token_counts[token] = token_counts.get(token, 0) + 1
                
                for token, count in token_counts.items():
                    logits[batch_idx, token] -= self.config.frequency_penalty * count
        
        return logits
    
    def _apply_penalties_beam(
        self,
        log_probs: torch.Tensor,
        generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica penalizaciones en beam search.
        
        Args:
            log_probs: (batch * beam_size, vocab_size)
            generated_tokens: (batch * beam_size, seq_len)
            
        Returns:
            log_probs modificados
        """
        if self.config.repetition_penalty == 1.0:
            return log_probs
        
        beam_batch_size = log_probs.size(0)
        
        for beam_idx in range(beam_batch_size):
            for token in set(generated_tokens[beam_idx].tolist()):
                if token == self.pad_token_id:
                    continue
                log_probs[beam_idx, token] /= self.config.repetition_penalty
        
        return log_probs


def calculate_coverage_penalty(
    attention_weights: torch.Tensor,
    coverage_penalty: float = 0.2
) -> torch.Tensor:
    """
    Calcula coverage penalty para evitar que el modelo ignore palabras del source.
    
    Args:
        attention_weights: (batch, n_heads, tgt_len, src_len)
        coverage_penalty: Factor de penalización
        
    Returns:
        Penalty scalar
    """
    # Sumar atención sobre todos los pasos de decodificación
    coverage = attention_weights.sum(dim=2)  # (batch, n_heads, src_len)
    
    # Penalizar posiciones con baja cobertura
    # log(coverage + epsilon) para evitar log(0)
    penalty = -torch.log(coverage + 1e-10).sum()
    
    return coverage_penalty * penalty
