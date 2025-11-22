# Guía de Implementación Técnica - Transformer NMT

## Resumen Ejecutivo

Este documento describe la implementación de un Transformer seq2seq para traducción automática neuronal (NMT) español-inglés, correspondiente a la Opción B de la Práctica Calificada 3.

## Arquitectura del Sistema

### Componentes Principales

#### 1. Encoder (Bidireccional)

El encoder procesa la secuencia de entrada (español) y genera representaciones contextuales.

**Flujo de datos:**
```
Input tokens → Embeddings → Positional Encoding → 
→ N × EncoderLayer → Output representations
```

**EncoderLayer:**
- Multi-Head Self-Attention bidireccional
- Layer Normalization + Residual Connection
- Feed-Forward Network (dim_feedforward = 4 × d_model)
- Layer Normalization + Residual Connection

**Características:**
- Atención sin restricciones: cada token puede atender a todos los demás
- Permite capturar dependencias de largo alcance
- Genera representación rica del contexto source

#### 2. Decoder (Autoregresivo)

El decoder genera la traducción token por token, condicionado en el encoder.

**Flujo de datos:**
```
Target tokens → Embeddings → Positional Encoding →
→ N × DecoderLayer → Linear projection → Logits
```

**DecoderLayer:**
- Masked Multi-Head Self-Attention (causal)
- Layer Normalization + Residual Connection
- Multi-Head Cross-Attention (al encoder)
- Layer Normalization + Residual Connection
- Feed-Forward Network
- Layer Normalization + Residual Connection

**Características clave:**
- Self-attention con máscara causal: token i solo ve tokens j ≤ i
- Cross-attention: conecta decoder con representaciones del encoder
- Generación autoregresiva: usa salidas previas como entrada

### Máscaras de Atención

#### Máscara Causal

Implementada como matriz triangular inferior:
```
[[True, False, False, False],
 [True, True,  False, False],
 [True, True,  True,  False],
 [True, True,  True,  True ]]
```

**Propósito:** Evitar que el modelo vea tokens futuros durante entrenamiento.

**Función:** `create_causal_mask(seq_len)`

**Test:** Verifica que `attn_weights[i,j] ≈ 0` si `j > i`

#### Máscara de Padding

Bloquea tokens de padding (PAD):
```
Secuencia: [token1, token2, PAD, PAD]
Máscara:   [True,   True,   False, False]
```

**Propósito:** Tokens de padding no contribuyen a la atención.

**Función:** `create_padding_mask(seq, pad_idx=0)`

**Test:** Verifica que suma de pesos de atención sobre padding ≈ 0

#### Máscara de Cross-Attention

Combina máscara de padding del source con longitud del target:
```
Shape: (batch, 1, tgt_len, src_len)
```

**Propósito:** Decoder solo atiende a tokens válidos del source.

### Codificaciones Posicionales

#### Sinusoidal (Original)

**Fórmula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Ventajas:**
- No requiere entrenamiento
- Permite extrapolación a longitudes mayores (limitada)

**Desventajas:**
- Rendimiento subóptimo en secuencias muy largas

#### RoPE (Rotary Position Embedding)

**Concepto:** Rota Q y K según la posición en vez de sumar vector posicional.

**Propiedades:**
- Preserva la norma: `||Q_rotated|| = ||Q||` (test crítico)
- Codifica información relativa de posición
- Mejor extrapolación que sinusoidal

**Interpolación NTK-aware:**
```
base_new = base × (α × target_len / original_len)^(dim / (dim - 2))
```

**Test:** `torch.allclose(q_norm_before, q_norm_after, atol=1e-5)`

#### ALiBi (Attention with Linear Biases)

**Fórmula:**
```
bias[i,j] = -m × |i - j|
```

donde `m` es la pendiente específica de cada cabeza.

**Ventajas:**
- Muy simple de implementar
- Excelente extrapolación a longitudes no vistas
- No añade parámetros

**Test:** Verifica monotonicidad estricta: `bias[i] ≥ bias[i+1]`

## Estrategias de Decodificación

### Greedy Search

**Algoritmo:**
```
for t in range(max_length):
    logits = model.decode_step(...)
    next_token = argmax(logits[-1])
    sequence.append(next_token)
```

**Características:**
- Determinista con misma seed
- Rápido (O(n))
- Puede quedar atrapado en óptimos locales

**Test:** Dos ejecuciones con misma seed → secuencias idénticas

### Beam Search

**Algoritmo:**
```
mantener k beams (hipótesis)
for cada paso:
    expandir cada beam con top-k tokens
    seleccionar k mejores hipótesis globales
    aplicar length penalty
retornar beam con mayor score normalizado
```

**Length Penalty:**
```
score_normalized = log_prob / (length^α)
```

donde `α ≈ 0.6` evita preferencia por secuencias cortas.

**Coverage Penalty:**
```
penalty = Σ log(min(attention_sum, 1.0))
```

Penaliza tokens del source que reciben poca atención.

**Características:**
- Mejor calidad que greedy (+3-5 BLEU típicamente)
- 5-10× más lento
- Requiere más memoria

### Top-k Sampling

**Algoritmo:**
```
logits = model.decode_step(...)
top_k_logits, top_k_indices = torch.topk(logits, k)
probs = softmax(top_k_logits / temperature)
next_token = sample(probs)
```

**Parámetros:**
- `k`: número de tokens candidatos (típicamente 50)
- `temperature`: controla aleatoriedad (1.0 = normal, >1 = más aleatorio)

**Uso:** Generación creativa, múltiples traducciones candidatas

### Top-p (Nucleus) Sampling

**Algoritmo:**
```
sorted_probs = sort(softmax(logits), descending=True)
cumsum = cumulative_sum(sorted_probs)
nucleus = tokens donde cumsum ≤ p
next_token = sample(nucleus)
```

**Parámetros:**
- `p`: probabilidad acumulada (típicamente 0.92)

**Ventaja sobre top-k:** Tamaño del núcleo se adapta a la distribución

### Penalizaciones

#### Repetition Penalty

```
if token in generated_tokens:
    logit[token] /= repetition_penalty
```

**Valor típico:** 1.2

**Propósito:** Evitar bucles ("the the the...")

#### Frequency Penalty

```
logit[token] -= frequency_penalty × count(token)
```

**Propósito:** Penalizar tokens que ya aparecieron múltiples veces

## Métricas de Evaluación

### Perplejidad

**Definición:**
```
PPL = exp(CrossEntropy)
```

**Interpretación:**
- PPL = 15: Bueno
- PPL = 30: Aceptable
- PPL = 100: Pobre

**Cálculo:**
```python
total_loss = 0
total_tokens = 0
for batch:
    logits = model(src, tgt_input)
    loss = CrossEntropyLoss(logits, tgt_output)
    total_loss += loss × n_tokens
    total_tokens += n_tokens
    
ppl = exp(total_loss / total_tokens)
```

### sacreBLEU

**Algoritmo:**
- Cuenta n-gramas coincidentes entre hipótesis y referencia
- Aplica penalización por brevedad
- Versión estandarizada con tokenización consistente

**Rango:** 0-100 (mayor es mejor)

**Comando:**
```python
import sacrebleu
bleu = sacrebleu.corpus_bleu(hypotheses, [references])
```

### chrF++

**Algoritmo:**
- Comparación a nivel de caracteres
- F-score sobre n-gramas de caracteres
- Mejor para idiomas con morfología rica

**Ventaja:** Captura similitudes morfológicas que BLEU puede perder

## Optimizaciones

### Mixed Precision Training (AMP)

**Implementación:**
```python
scaler = GradScaler()

with autocast():
    logits = model(src, tgt)
    loss = criterion(logits, target)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

**Beneficios:**
- 2× velocidad de entrenamiento
- 2× reducción de memoria
- Precisión similar a FP32

### KV-Cache

**Propósito:** Evitar recalcular atención para tokens ya generados.

**Implementación:**
```python
class KVCache:
    def __init__(self, max_batch, max_seq, n_heads, head_dim):
        self.cache_k = zeros(max_batch, n_heads, max_seq, head_dim)
        self.cache_v = zeros(max_batch, n_heads, max_seq, head_dim)
    
    def update(self, k, v, start_pos):
        self.cache_k[:, :, start_pos:start_pos+len, :] = k
        self.cache_v[:, :, start_pos:start_pos+len, :] = v
        return self.cache_k[:, :, :start_pos+len, :], ...
```

**Speedup:** 5-10× en generación de secuencias largas

### Learning Rate Schedule

**Warmup + Cosine Decay:**
```python
if step < warmup_steps:
    lr = base_lr × (step / warmup_steps)
else:
    progress = (step - warmup) / (total - warmup)
    lr = min_lr + (base_lr - min_lr) × 0.5 × (1 + cos(π × progress))
```

**Parámetros típicos:**
- warmup_steps: 1000-4000
- base_lr: 1e-4
- min_lr: 1e-6

## Tests Críticos

### Test de Máscaras

**test_causal_mask_values:**
```python
mask = create_causal_mask(4)
assert mask[0, 1] == False  # Futuro bloqueado
assert mask[2, 1] == True   # Pasado permitido
```

**test_padding_mask:**
```python
seq = [[1, 2, 0, 0]]  # 2 tokens válidos, 2 padding
mask = create_padding_mask(seq)
assert mask[0, 0, 0, 2] == False  # Padding bloqueado
```

### Test de RoPE

**test_norm_preservation:**
```python
q_norm_before = q.norm(dim=-1)
q_rotated, _ = rope(q, k)
q_norm_after = q_rotated.norm(dim=-1)
assert torch.allclose(q_norm_before, q_norm_after, atol=1e-5)
```

### Test de ALiBi

**test_monotonicity:**
```python
bias = alibi(seq_len)
for i in range(seq_len - 1):
    assert bias[0, head, 0, i] >= bias[0, head, 0, i+1]
```

### Test de Decodificación

**test_greedy_deterministic:**
```python
set_seed(42)
output1 = decoder.greedy_search(src)
set_seed(42)
output2 = decoder.greedy_search(src)
assert output1 == output2
```

## Estructura de Archivos

### Modelos (`src/models/`)

**attention.py:**
- `ScaledDotProductAttention`: Implementación manual de atención
- `create_causal_mask`, `create_padding_mask`: Funciones de máscaras
- `KVCache`: Cache para generación eficiente

**posenc.py:**
- `SinusoidalPositionalEncoding`: Codificación sinusoidal
- `RotaryPositionalEmbedding`: RoPE con interpolación
- `ALiBiPositionalBias`: Sesgos lineales por cabeza

**mhsa.py:**
- `MultiHeadAttention`: Atención multi-cabeza genérica
- `MultiHeadSelfAttention`: Self-attention (Q=K=V)
- `MultiHeadCrossAttention`: Cross-attention (Q≠K=V)
- `FeedForwardNetwork`: FFN con ReLU

**transformer.py:**
- `TransformerConfig`: Dataclass de configuración
- `EncoderLayer`, `DecoderLayer`: Capas individuales
- `Encoder`, `Decoder`: Stacks de capas
- `Transformer`: Modelo completo

### Utilidades (`src/`)

**data.py:**
- `Vocabulary`: Manejo de vocabulario con tokens especiales
- `TranslationDataset`: Dataset para pares de traducción
- `download_tatoeba`: Descarga desde Hugging Face
- `collate_fn`: Padding dinámico para batches

**train.py:**
- `Trainer`: Clase principal de entrenamiento
- `WarmupCosineScheduler`: LR schedule
- Soporte para AMP, gradient clipping, early stopping

**eval.py:**
- `Evaluator`: Evaluación con múltiples métricas
- Cálculo de perplejidad, BLEU, chrF++
- Generación de muestras para inspección

**decoding.py:**
- `Decoder`: Clase con todas las estrategias
- Implementación de greedy, beam, top-k, top-p
- Aplicación de penalizaciones

**utils.py:**
- `set_seed`: Reproducibilidad completa
- `save_checkpoint`, `load_checkpoint`: Persistencia
- `get_device`: Auto-detección de hardware
- `AverageMeter`: Tracking de métricas

## Comandos de Uso

### Preparar Datos
```bash
python -m src.data --prepare --config configs/train.yaml
```

### Entrenar
```bash
python -m src.train --config configs/train.yaml
```

### Evaluar
```bash
python -m src.eval --checkpoint checkpoints/best_model.pt --split test
```

### Generar Traducciones
```bash
python -m src.decoding --strategy beam --beam_size 4
```

### Ejecutar Tests
```bash
pytest tests/ -v
```

### Profiling
```bash
python -m src.train --config configs/train.yaml --profile
```

## Notas de Implementación

### Reproducibilidad

Todas las fuentes de aleatoriedad están controladas:
```python
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Gestión de Memoria

Para secuencias largas:
- Usar gradient checkpointing
- Reducir batch_size
- Activar KV-cache paginado

### Debugging

Verificar shapes en cada paso:
```python
print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
print(f"attn_output: {attn_output.shape}")
```

Inspeccionar pesos de atención:
```python
output, attn_weights = mhsa(x, return_attention=True)
plt.imshow(attn_weights[0, 0].detach())
```

## Referencias Técnicas

- Vaswani et al. (2017): "Attention Is All You Need" - Arquitectura base
- Su et al. (2021): "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Press et al. (2021): "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
- Karpathy (2022): "nanoGPT" - Implementación de referencia
