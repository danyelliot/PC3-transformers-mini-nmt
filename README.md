# Transformer NMT - Práctica Calificada 3 (Opción B)

## Descripción

Implementación completa de un Transformer seq2seq para traducción automática neuronal (NMT) español-inglés, desarrollado como parte de la Práctica Calificada 3 del curso CC0C2.

### Características Principales

- **Encoder bidireccional** con 4-6 capas de Multi-Head Self-Attention
- **Decoder autoregresivo** con self-attention causal y cross-attention
- **Múltiples codificaciones posicionales**: Sinusoidal, RoPE, ALiBi
- **Estrategias de decodificación avanzadas**:
  - Greedy search
  - Beam search con length penalty y coverage penalty
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Penalizaciones por repetición y frecuencia
- **Optimizaciones**: Mixed Precision (AMP), Gradient Checkpointing, KV-cache
- **Tests unitarios** completos con pytest
- **Profiling** detallado con torch.profiler

## Arquitectura

```
Transformer NMT
├── Encoder
│   ├── Token Embedding + Positional Encoding
│   └── N × EncoderLayer
│       ├── Multi-Head Self-Attention (bidireccional)
│       ├── Layer Norm + Residual
│       ├── Feed-Forward Network
│       └── Layer Norm + Residual
│
└── Decoder
    ├── Token Embedding + Positional Encoding
    └── N × DecoderLayer
        ├── Masked Multi-Head Self-Attention (causal)
        ├── Layer Norm + Residual
        ├── Multi-Head Cross-Attention (→ Encoder)
        ├── Layer Norm + Residual
        ├── Feed-Forward Network
        ├── Layer Norm + Residual
        └── Linear Projection → Vocabulario
```

## Estructura del Proyecto

```
pc3-transformers/
├── Makefile                    # Comandos de make
├── Dockerfile                  # Contenedor Docker
├── requirements.txt            # Dependencias Python
├── .gitignore                 # Archivos a ignorar
├── README.md                  # Este archivo
│
├── configs/
│   └── train.yaml             # Configuración de entrenamiento
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Utilidades (seeds, checkpoints, logging)
│   ├── data.py                # Carga y procesamiento de datos
│   ├── train.py               # Loop de entrenamiento
│   ├── eval.py                # Evaluación y métricas
│   ├── decoding.py            # Estrategias de decodificación
│   │
│   └── models/
│       ├── __init__.py
│       ├── attention.py       # SDPA + máscaras + KV-cache
│       ├── posenc.py          # Sinusoidal, RoPE, ALiBi
│       ├── mhsa.py            # Multi-Head Attention
│       └── transformer.py     # Encoder, Decoder, Transformer
│
├── tests/
│   ├── test_attention.py      # Tests de atención y máscaras
│   ├── test_posenc.py         # Tests de codificaciones posicionales
│   ├── test_shapes.py         # Tests de dimensiones
│   └── test_decoding.py       # Tests de decodificación
│
├── notebooks/
│   ├── 01_teoria.ipynb        # Explicación teórica
│   ├── 02_entrenamiento.ipynb # Experimentos de entrenamiento
│   ├── 03_decoding.ipynb      # Comparación de estrategias
│   └── 04_eval_perf.ipynb     # Profiling y métricas
│
├── data/                      # Datos (no versionados)
│   ├── raw/
│   └── processed/
│
├── checkpoints/               # Checkpoints del modelo
├── outputs/                   # Salidas y logs
└── logs/                      # Archivos de log
```

## Instalación y Uso

### Opción 1: Docker (Recomendado)

```bash
# Construir la imagen
docker build -t pc3-transformer .

# Ejecutar con Jupyter
docker run --rm -p 8888:8888 -v $PWD:/app pc3-transformer

# Ejecutar entrenamiento
docker run --rm -v $PWD:/app pc3-transformer bash -lc "make setup && make train && make eval"
```

### Opción 2: Instalación Local

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
make setup

# O manualmente
pip install -r requirements.txt
```

## Flujo de Trabajo

### 1. Preparar Datos

```bash
make data
```

Esto descargará y procesará el dataset Tatoeba español-inglés (~20k pares).

### 2. Entrenar el Modelo

```bash
# Entrenamiento básico
make train

# Con profiling
make profile
```

La configuración se encuentra en `configs/train.yaml`.

### 3. Evaluar

```bash
make eval
```

Calcula métricas: sacreBLEU, chrF++, perplejidad.

### 4. Generar Traducciones

```bash
# Beam search (default)
make decode

# Greedy search
python -m src.decoding --strategy greedy

# Top-p sampling
python -m src.decoding --strategy topp --top_p 0.92
```

### 5. Ejecutar Tests

```bash
make test
```

Ejecuta todos los tests unitarios con pytest.

## Tests Implementados

### Máscaras
- Máscara causal: token i no puede atender a j > i
- Máscara de padding: tokens PAD no aportan ni reciben atención
- Combinación de máscaras

### Codificaciones Posicionales
- RoPE: Preservación de norma (tolerancia 1e-6)
- ALiBi: Monotonicidad estricta en el sesgo
- Interpolación para longitudes mayores

### Decodificación
- Greedy: Determinismo con misma seed
- Beam search: Length penalty favorece secuencias coherentes
- Top-k/Top-p: Diversidad medida con self-BLEU

### Shapes
- Dimensiones correctas en todas las capas
- Atención multi-cabeza: (batch, n_heads, seq_len, head_dim)
- Cross-attention: query del decoder, key/value del encoder

## Métricas y Evaluación

### Métricas Principales

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **sacreBLEU** | BLEU case-sensitive con tokenización estándar | Maximizar |
| **chrF++** | Character n-gram F-score (mejor para morfología) | Maximizar |
| **Perplejidad** | exp(cross-entropy) en validación | Minimizar |
| **Tokens/s** | Velocidad de generación | Maximizar |
| **Peak VRAM** | Memoria GPU máxima usada | Minimizar |

### Tabla de Resultados Esperados

| Config | BLEU | chrF++ | PPL | Tokens/s | VRAM |
|--------|------|--------|-----|----------|------|
| Base (Sinusoidal) | ~18-22 | ~45-50 | 15-20 | 280 | 2.1 GB |
| + RoPE | ~20-24 | ~47-52 | 13-18 | 290 | 2.0 GB |
| + Beam Search (k=4) | ~22-26 | ~50-54 | - | 50 | 2.8 GB |

## Configuración del Modelo

Edita `configs/train.yaml`:

```yaml
model:
  d_model: 256              # Dimensión del modelo
  nhead: 8                  # Número de cabezas
  num_encoder_layers: 4     # Capas encoder
  num_decoder_layers: 4     # Capas decoder
  dim_feedforward: 1024     # Dimensión FFN
  dropout: 0.1              # Dropout
  pos_encoding: "sinusoidal"  # sinusoidal, rope, alibi

training:
  epochs: 30
  learning_rate: 0.0001
  batch_size: 32
  warmup_steps: 1000
  label_smoothing: 0.1
  use_amp: true             # Mixed precision

decoding:
  strategy: "beam"
  beam_size: 4
  length_penalty: 0.6       # α en length^α
  coverage_penalty: 0.2
  repetition_penalty: 1.2
```

## Profiling

El profiling se realiza con `torch.profiler`:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Forward pass o generación
    ...

# Ver resultados
print(prof.key_averages().table(sort_by="cuda_time_total"))

# Exportar para visualización
prof.export_chrome_trace("trace.json")
# Abrir en chrome://tracing
```

### Escenarios a Profiling

1. **SDPA manual vs torch.sdpa** (FlashAttention)
2. **Multi-Head Attention completo**
3. **Feed-Forward Network**
4. **Greedy vs Beam Search** (generación completa)

## Referencias

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [RoPE: Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [ALiBi: Attention with Linear Biases (Press et al., 2021)](https://arxiv.org/abs/2108.12409)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

## Autor

Desarrollado para el curso CC0C2 - Ciencia de la Computación

## Licencia

MIT License - Uso educativo

## Troubleshooting

### Error: OOM (Out of Memory)

```yaml
# Reducir batch_size en configs/train.yaml
training:
  batch_size: 16  # Era 32
```

### Error: CUDA not available

El código funciona en CPU. Para forzar CPU:

```yaml
system:
  device: "cpu"
```

### Tests fallan con diferencias numéricas

```python
# Verificar que las semillas están fijadas
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Criterios de Evaluación

- Notebook (6 pts): Teoría + experimentos + gráficos
- Video (4 pts): Demo + métricas + conclusiones (máximo 8 minutos)
- Exposición (10 pts): Arquitectura + ablaciones + preguntas (5-7 minutos)

## Estado del Proyecto

Implementación completa de los componentes core:
- Modelos: attention.py, posenc.py, mhsa.py, transformer.py
- Datos: data.py con descarga y procesamiento de Tatoeba
- Entrenamiento: train.py con warmup, cosine decay, AMP
- Evaluación: eval.py con sacreBLEU, chrF++, perplejidad
- Decodificación: decoding.py con greedy, beam, top-k, top-p
- Tests: test_attention.py, test_posenc.py, test_shapes.py, test_decoding.py

Pendiente:
- Notebooks de análisis y visualización
- Ejecución de entrenamiento completo
- Generación de gráficos comparativos
- Video demostrativo
- Presentación para exposición
