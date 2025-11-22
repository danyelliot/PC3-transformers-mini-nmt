## Práctica calificada 3-CC0C2 

#### Instrucciones generales

- **Plazo**: 6 días. Entrega el 22 de noviembre, en hora de clases.
- **Tiempo estimado**: ~6 h (3 h implementación, 1.5 h teoría, 1 h video, 0.5 h exposición)  
- **Formato**: Trabajo individual  
- **Tecnologías obligatorias**: Python (PyTorch), Docker, Makefile, Jupyter Notebooks, pytest, torch.profiler  
- **Ejecución local-first**: no se requiere GPU; si tienes CUDA, puedes activarla opcionalmente  
- **Datos**: usar **toy datasets** incluidos o descargables pequeños (~ 1-20 MB)  

#### Objetivo

El objetivo principal es diseñar, entrenar y analizar exhaustivamente un **micro-Transformer** (parámetros entre 1M y 30M) que sirva como laboratorio vivo para experimentar con los componentes críticos de los modelos modernos. No solo implementaremos cada pieza, sino que las 
aislaremos para medir su impacto real en calidad, velocidad y consumo de recursos. 

Los tres pilares se evaluarán de forma cruzada: por ejemplo, ¿cómo cambia la perplejidad si pasamos de Sinusoidal a RoPE con NTK-aware en un contexto de 8k tokens? ¿Cuánta VRAM ahorra realmente un KV-cache paginado de 256 tokens por bloque frente a uno continuo?.

El proyecto final debe incluir gráficos comparativos claros (curvas de pérdida, histogramas de atención, tablas de Pareto latencia/calidad) y conclusiones accionables que podrías citar en una entrevista técnica.

### Alcance y opciones de proyecto (elige una)

**Opción A - Núcleo de Atención + Posiciones (recomendada si quieres dominar los fundamentos desde cero)**  
Aquí construyes solo el decoder (como GPT) pero con máxima transparencia.  
- **Scaled Dot-Product Attention**: implementas manualmente la fórmula de atención con máscara causal (triángulo inferior), máscara de padding (para batches desiguales) y una máscara selectiva que puedes activar/desactivar para pruebas de forma. Esto te permite imprimir los tensores Q, K, V y ver exactamente cómo se alinean los shapes.  
- **MHSA completo**: cada cabecera tiene su propia proyección lineal; al final del entrenamiento puedes visualizar 8 mapas de atención distintos y descubrir cuál cabecera aprendió a atender a la palabra anterior, cuál a puntuación, etc.  
- **Codificaciones posicionales comparadas**:  
  - **Sinusoidal clásico**: la original del paper "Attention is All You Need". Fija, no aprendible.  
  - **RoPE** (Rotary Positional Embedding): en vez de sumar un vector, rota las consultas y claves según la posición. Incluye soporte para **interpolación**: NTK-aware (ajusta dinámicamente la frecuencia base según la longitud deseada), Position Interpolation (estira linealmente las posiciones) y YaRN (método de Meta que combina varios trucos).  
  - **ALiBi**: no usa embeddings, simplemente resta un valor negativo que crece linealmente con la distancia |i-j|. Muy simple y sorprendentemente efectivo en extrapolación.  
- **KV-cache corta/paginada**: durante generación guardas solo bloques de 256 tokens; cuando se llena el bloque, lo descargas a CPU o lo comprimes. Mides **Peak VRAM** real con `torch.cuda.max_memory_allocated()`.  
- **Tarea**: TinyShakespeare o un corpus de 500-1000 cuentos cortos en español (puedes descargarlos de Proyecto Gutenberg).  
- **Métricas clave**:  
  - **Cross-entropy y Perplejidad (PPL)**: menor es mejor.  
  - **Tokens/s** en generación (mide con batch_size=1, seq_len=2048).  
  - **Peak VRAM / RSS**: gráfico de cómo crece con longitud de contexto.  
- **Ventaja práctica**: terminas con un notebook donde puedes cambiar una línea (RoPE -> ALiBi) y ver instantáneamente el efecto en atención y memoria.

**Opción B - Mini NMT (Encoder-Decoder) + Decodificación**  
Construyes un Transformer completo seq2seq como el original de 2017 pero a escala juguete.  
- **Encoder**: 4-6 capas de MHSA bidireccional + FFN.  
- **Decoder**: self-attention causal + cross-attention sobre salidas del encoder (con máscara de padding).  
- **Decodificación avanzada**:  
  - **Beam search**: anchura 4-8, con **length penalty** (explicación: divide el log-prob total por longitud^α, donde α≈0.6; evita que el modelo prefiera frases cortas solo porque acumulan menos probabilidad negativa).  
  - **Coverage penalty**: resta un término que penaliza cuando algún token del source no ha sido atendido suficientemente; ayuda a no ignorar palabras.  
  - Comparas con **top-k** (k=50), **top-p (nucleus)** (p=0.92) y **temperatura** (0.7-1.2).  
  - **Repetition penalty 1.2**: si un token ya apareció, su logit se divide por 1.2, evita bucles infinitos tipo "the the the".  
  - **Frequency penalty**: resta un valor proporcional a cuántas veces apareció el token en el output actual.  
- **Datos**: pares español-inglés de Tatoeba (filtras a 20k).  
- **Métricas**: sacreBLEU, chrF++, velocidad de inferencia.  
- **Ventaja**: ves en práctica real cómo beam search con length penalty sube +3-5 BLEU pero es 5-10× más lento.

**Opción C - Clasificación con Encoder-Only + Prompting Ligero**  
Modelo BERT pequeño para clasificación de texto.  
- Pooling con token `[CLS]`.  
- Comparas:  
  - Fine-tuning completo (todos los parámetros).  
  - **Soft-prompts**: añades 20-100 tokens aprendibles al inicio y congelas el resto.  
  - LoRA rank 8 solo en matrices de atención.  
- **Ventaja**: aprendes técnicas que usan empresas cuando no pueden entrenar todo el modelo.

**Opción D - Atención eficiente y throughput**  
Aquí el foco es velocidad y memoria.  
- Comparas `torch.nn.functional.scaled_dot_product_attention` (FlashAttention-2 interno) vs tu implementación manual.  
- Pruebas **sliding-window** de 512 tokens y **block-sparse**.  
- Usas `torch.profiler` para obtener **tokens/s**, **ms por batch**, **Peak VRAM**.  
- Activas **gradient checkpointing** y BF16.  
- Graficas curvas de Pareto: calidad (PPL) vs latencia.

**Opción E - Largo contexto con RoPE/ALiBi + Interpolación y KV-cache**  
Especialización en contextos largos.  
- **NTK-aware dynamic scaling**: al ver una longitud mayor a la de entrenamiento, reduces la frecuencia base de RoPE con fórmula logarítmica para mantener la distribución de ángulos.  
- **Needle-in-a-haystack**: escondes una frase clave (la "aguja") en posición aleatoria dentro de 8k tokens de relleno y mides **recall@distancia** (porcentaje de veces que el modelo la recupera).  
- KV-cache paginada con bloques de 512; cuando se llena, mueves bloques antiguos a CPU y los recargas solo si son necesarios.  
- Mides PPL por tramos (0-1k, 1k-4k, 4k-8k) y haces gráfico de degradación.

**Opción F - MoE ligero / Adapters + Routing y costo**  
Implementas Mixture-of-Experts dentro del FFN.  
- 2-4 expertos por capa, gating top-2 (el token usa los dos expertos con mayor score).  
- Mides:  
  - **Histograma de scores de gating** (para ver si algún experto está muerto).  
  - % de tokens que van a cada experto (balance de carga).  
  - FLOPs reales por token (MoE puede ser más barato que un FFN denso grande).  
- **Ventaja**: entiendes por qué Grok-1 y Mixtral 8x7B son rápidos a pesar de tener tantos parámetros.


#### Entregables
Repositorio con esta estructura mínima:

```text
pc3-transformers/
├─ Makefile
├─ Dockerfile
├─ requirements.txt
├─ configs/
│  └─ train.yaml
├─ src/
│  ├─ models/
│  │  ├─ attention.py        # SDPA + máscaras + KV-cache
│  │  ├─ mhsa.py             # Multi-head + proyecciones
│  │  ├─ posenc.py           # Sinusoidal, RoPE, ALiBi, interpolación
│  │  └─ transformer.py      # Bloques: Attn -> FFN -> Norm/Residual/Dropout
│  ├─ decoding.py            # greedy, beam, top-k/p, temperatura, penalizaciones
│  ├─ data.py                # carga/limpieza, batching, padding masks
│  ├─ train.py               # loop CLM/MLM/seq2seq (según opción)
│  ├─ eval.py                # métricas (PPL/BLEU/chrF/F1)
│  └─ utils.py               # seeds, logging, checkpoints
├─ tests/
│  ├─ test_masks.py          # causal/padding/selectiva
│  ├─ test_posenc.py         # RoPE/ALiBi propiedades
│  ├─ test_decoding.py       # determinismo, penalizaciones
│  └─ test_shapes.py         # shapes Q/K/V y heads
├─ notebooks/
│  ├─ 01_teoria.ipynb        # bloques, pérdidas, escalado
│  ├─ 02_entrenamiento.ipynb # entrenamiento + logs
│  ├─ 03_decoding.ipynb      # greedy/beam/top-k/p
│  └─ 04_eval_perf.ipynb     # métricas y profiling (torch.profiler)
├─ data/                     # corpus mini (o script de descarga)
└─ README.md
```

- Notebooks ejecutables (celdas limpias, `seed=42`).  
- Video ≤8 min (guion tipo sprint, demo, métricas, cierre).  
- Exposición 5-7 min (internals + preguntas).

#### Makefile (mínimo sugerido)

```make
PY=python

.PHONY: setup data train eval decode profile test clean

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data --prepare

train:
	$(PY) -m src.train --config configs/train.yaml

eval:
	$(PY) -m src.eval --split val

decode:
	$(PY) -m src.decoding --strategy beam --beam_size 4 --length_penalty 0.7

profile:
	$(PY) -m src.train --config configs/train.yaml --profile

test:
	pytest -q

clean:
	rm -rf outputs/ checkpoints/ .pytest_cache
```

#### Dockerfile (base CPU; CUDA opcional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
CMD ["bash", "-lc", "make setup && make test && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root"]
```

#### Ejecución típica

```bash
docker build -t pc3 .
docker run --rm -p 8888:8888 -v $PWD:/app pc3
# Sin notebook:
docker run --rm pc3 bash -lc "make setup && make train && make eval"
```

#### Requisitos de pruebas (obligatorio)

- **Máscaras** (tienes que implementarlas tú, no usar `nn.MultiheadAttention` que las oculta):  
  - **Causal**: matriz triangular superior con `-inf` arriba de la diagonal. En generación autoregresiva, el token 5 nunca puede atender al token 6. Test: genera una secuencia y verifica que `attn_weights[i,j] = 0` si `j > i`.  
  - **Padding**: donde el token es `PAD=0`, pones `-inf` en toda la fila de K (o columna de Q). Así el padding nunca aporta ni recibe atención. Test: mete un batch con longitudes [10, 7, 7, 4] y comprueba que la pérdida ignora los pads (usa `ignore_index=0` en CrossEntropyLoss).  
  - **Selectiva**: una máscara booleana extra que puedes usar para anular posiciones específicas (útil para ablation: "¿qué pasa si le quito atención a los artículos?"). Test: crea una máscara que bloquee todos los tokens de tipo "DET" y mide caída de PPL.

- **Posiciones** (deben pasar tests automáticos):  
  - **RoPE**: después de aplicar rotación, la norma de Q y K debe ser idéntica al vector original (±1e-6). Implementa `apply_rotary_emb` y testea `torch.allclose(q.norm(dim=-1), q_original.norm(dim=-1))`.  
  - **Interpolación de RoPE**: al pasar de contexto 2048 -> 8192, no puedes romper shapes. Prueba NTK-aware (cambia `base` dinámicamente), Position Interpolation (mapea pos 0-8191 -> 0-2047) y YaRN (combina ambos). Test: genera 8192 tokens sin OOM y sin NaN.  
  - **ALiBi**: el sesgo es `-m * (i - j)` con `m > 0`. Test de monotonicidad: para una cabecera fija, `attn_bias[0, 100] < attn_bias[0, 50] < attn_bias[0, 0]`. Si no es estrictamente decreciente, falla el test.

- **Decodificación** (todas deben ser deterministas con `torch.manual_seed(42)`):  
  - **Greedy**: siempre el argmax. Dos corridas con misma seed -> misma secuencia 100 %.  
  - **Beam search**: con `length_penalty=0.6` una secuencia de 20 tokens debe ganar frente a una de 10 tokens aunque tenga logprob ligeramente menor. Test: desactiva penalty -> prefiere corto; actívalo -> prefiere largo pero coherente.  
  - **Top-k / top-p**: genera 50 secuencias y mide diversidad (self-BLEU bajo = más diverso). Curva: k=1 -> greedy, k=50 -> muy diverso, p=0.95 suele ser sweet spot.

- **Reproducibilidad total**:  
  - `torch.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.  
  - Tres corridas independientes -> PPL media con desviación ≤1 % (ej: 18.3 ± 0.2).

#### Profiling (obligatorio)

Usa este bloque exacto en tu código:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # tu forward o generate aquí

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")  # ábrelo en chrome://tracing
```

Mide **cuatro escenarios** como mínimo:  
1. **SDPA manual** vs **torch.sdpa** (FlashAttention) -> espera 3-8× speedup y 40 % menos VRAM.  
2. **MHSA completo** (QKV proj + atención + Out proj).  
3. **FFN** (o MoE): aquí suele estar el 60 % del tiempo en modelos pequeños.  
4. **Decoding**: greedy 2048 tokens vs beam=4 -> beam es 5-12× más lento.

Reporta siempre:  

- **Tokens/s** (generación, batch=1, seq=2048).  
- **ms/batch** (entrenamiento, batch=8).  
- **Peak VRAM** con `torch.cuda.max_memory_allocated() / 1e9` GB.  
- **RSS** si corres en CPU-only (para laptops).

Compara en una tabla:  

| Config                  | PPL   | Tokens/s | Peak VRAM | ms/batch |
|-------------------------|-------|----------|-----------|----------|
| Sinusoidal + sin cache  | 22.1  | 280      | 4.1 GB    | 180      |
| RoPE + KV-cache 256     | 19.8  | 620      | 1.9 GB    | 92       |
| ALiBi + cache paginado  | 20.3  | 590      | 1.7 GB    | 95       |

#### Métricas y evaluación 

- **Modelado de lenguaje**:  
  - Cross-entropy en validación.  
  - Perplejidad = exp(cross-entropy). Gráfico de PPL vs época con error bars (shade = ±std de 3 seeds).  
- **Traducción**: sacreBLEU (case-sensitive), chrF++ (mejor con español por morfología).  
- **Clasificación**: accuracy y F1-macro. Tabla con 3 seeds + promedio.  
- **Gráficos obligatorios**:  
  - Curva de pérdida train/val.  
  - Pareto latencia vs PPL.  
  - Histograma de atención promedio por capa.  
  - Barras de velocidad decoding (greedy/beam/top-p).

#### Rúbrica (20 puntos) 

- **Notebook (6 pts)**:  
  - Una celda `!nvidia-smi` y `!cat /proc/meminfo` al inicio.  
  - Sección "Teoría" con 1-2 párrafos por componente (cita papers).  
  - `%%time` en cada experimento.  
- **Video (4 pts)**:  
  - Guion sprint: 30 s intro, 2 min demo generación, 1 min gráficos, 30 s conclusiones.  
  - Graba pantalla + voz + cara pequeña (trust + claridad).  
  - Sube sin editar a YouTube privado o Drive con enlace.  
- **Exposición oral (10 pts)**:  
  - Slides: título, objetivo, arquitectura, 2 slides de resultados, 1 de ablaciones, 1 de conclusiones.  
  - Practica responder: "¿Por qué RoPE supera a Sinusoidal en 8k?", "¿Cuánta VRAM ahorra FlashAttention?", "¿Qué pasa si quito length_penalty?".

#### Cronograma recomendado (6 días)

- **Día 1**:  
  - Repo con `Dockerfile` (python:3.11-slim + torch 2.4 + cuda 12 opcional).  
  - `make setup`, `make test` pasa (aunque solo smoke tests).  
  - Descarga datos y guarda en `data/raw/`.  
- **Día 2**:  
  - `attention.py` con tres máscaras + tests unitarios (`pytest tests/test_attention.py`).  
  - Visualiza máscara causal con `plt.imshow`.  
- **Día 3**:  
  - `posenc.py`: tres clases heredando de `nn.Module`.  
  - Test de norma RoPE y monotonicidad ALiBi.  
  - Mini-experimento: entrena 100 steps con cada una y guarda PPL.  
- **Día 4**:  
  - `transformer.py` (Config dataclass, bloques, LayerNorm PRE).  
  - `train.py` con warmup 100 steps, cosine decay, grad-clip 1.0, label smoothing 0.1.  
  - AMP (`torch.amp.autocast('cuda')` y `GradScaler`).  
- **Día 5**:  
  - `decoding.py`: clase `Decoder` con greedy, beam, topk, topp.  
  - `profile.py` con los 4 escenarios.  
  - Genera 10 ejemplos con cada estrategia y guarda en `samples/`.  
- **Día 6**:  
  - Notebook final: carga checkpoints, tablas automáticas con `pandas`, gráficos con `seaborn`.  
  - Graba video (máx 5 min).  
  - `README.md` con gifs de atención y comandos exactos.

#### Reglas y buenas prácticas

- **Curación mínima**: NFKC Unicode, minúsculas, quita duplicados exactos, filtra secuencias > 1024 tokens.  
- **Split sin leaks**: estratificado por longitud o autor.  
- **Regularización**: dropout 0.1, label smoothing 0.1, weight decay 0.01, early stopping patience 5.  
- **Ablations honestos**: tabla "sin máscara causal -> PPL explota", "sin KV-cache -> OOM a 4k".  
- **Sesgos**: si usas cuentos clásicos, menciona "predominan personajes europeos, siglo XIX".

#### Criterio mínimo de aprobación (MVP) 

- Modelo 8M params que genera 100 tokens coherentes.  
- RoPE + ALiBi funcionando.  
- Greedy + Beam search (con length_penalty=0.6).  
- Tabla de profiling con 3 filas.  
- `pytest -v` -> 15/15 passed.  
- Repo con Docker que corre `make train && make decode`.

#### Ideas de extras (para llegar a 20/20 y destacar)

- **Gradient checkpointing** + BF16 -> entrena en RTX 3060 12 GB con contexto 16k.  
- **FlashAttention real** (`pip install flash-attn`) -> +300 % tokens/s.  
- **LoRA rank 8** + **soft-prompts** de 32 tokens -> fine-tuning en 5 min.  
- **KV-cache paginada** con `torch.storage` en CPU -> contexto 32k en 8 GB VRAM.  
- **Curva de escalamiento**: entrena 1M, 4M, 12M, 30M -> gráfico log-log PPL vs params.  
- **CFG (Classifier-Free Guidance)** en generación: interpolar entre condicional y incondicional.



#### Datasets mini sugeridos – enlaces directos

* **LM**:

  * TinyShakespeare: `wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
    -> [Abrir TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
  * Cuentos español: `wget https://raw.githubusercontent.com/josefernandezamor/cuentos-populares-españoles/master/cuentos.txt`
    -> *(el enlace original puede fallar)*
    **Alternativas (TXT UTF-8, dominio público):**

    * `wget https://www.gutenberg.org/ebooks/46196.txt.utf-8`  *(Cuentos clásicos del norte-1)*
      -> [Abrir 46196.txt.utf-8](https://www.gutenberg.org/ebooks/46196.txt.utf-8)
    * `wget https://www.gutenberg.org/ebooks/46496.txt.utf-8`  *(Cuentos clásicos del norte-2)*
      -> [Abrir 46496.txt.utf-8](https://www.gutenberg.org/ebooks/46496.txt.utf-8)
    * `wget https://www.gutenberg.org/ebooks/55514.txt.utf-8`  *(Pardo Bazán - Cuentos de amor)*
      -> [Abrir 55514.txt.utf-8](https://www.gutenberg.org/ebooks/55514.txt.utf-8)

* **Traducción**: Tatoeba en-es 20k -> `wget https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-17/txt/es-en.txt.gz`
  -> *(el enlace original puede fallar)*
  
  **Alternativa A (recomendada, sin URLs frágiles - Hugging Face Datasets):**

  ```bash
  python - << 'PY'
  from datasets import load_dataset
  ds = load_dataset("Helsinki-NLP/tatoeba", lang1="spa", lang2="eng", split="test")
  ds = ds.select(range(min(20000, len(ds))))  # recorta a ~20k
  with open("tatoeba.es","w",encoding="utf-8") as es, open("tatoeba.en","w",encoding="utf-8") as en:
      for ex in ds:
          es.write((ex.get("sentence1") or "").strip().replace("\t"," ")+"\n")
          en.write((ex.get("sentence2") or "").strip().replace("\t"," ")+"\n")

  ```

  **Alternativa B (línea de comando - OPUS Tools):**

  ```bash
  pip install -U opustools-pkg
  opus_get -s spa -t eng Tatoeba -o tatoeba-es-en.zip
  unzip -p tatoeba-es-en.zip Tatoeba/spa-eng.txt > tatoeba.es-en.txt || true
  ```

* **Clasificación**: SST-2 2k -> `pip install datasets && python -c "from datasets import load_dataset; d=load_dataset('sst2'); d['train'].select(range(2000)).save_to_disk('sst2_mini')"`
  -> *(tu una-liner carga mal SST-2; versión corregida abajo)*
  **Alternativa corregida (GLUE/SST-2):**

  ```bash
  pip install -U datasets
  python - << 'PY'
  from datasets import load_dataset, load_from_disk
  ds = load_dataset("glue", "sst2")
  ds_small = ds["train"].select(range(2000))
  ds_small.save_to_disk("sst2_mini")
  # verificación de carga posterior:
  _ = load_from_disk("sst2_mini")
  print("SST-2 mini listo (2000 ejemplos).")
  PY
  ```

#### Entrega 

-  `docker build -t micro-transformer .` -> OK  
-  `docker run --gpus all micro-transformer make test` -> 100 % pass  
-  `make train` -> termina en <4 h en Colab free o laptop  
-  `make decode` -> genera 5 ejemplos bonitos  
-  `make profile` -> genera `trace.json` y tabla markdown  
-  Notebook con todas las tablas/figuras (no celdas rotas)  
-  Video subido (enlace en README)  
-  `README.md` con:  
  ```bash
  make setup && make train && make decode && make profile
  ```
-  `.gitignore` correcto (no subas `*.pth`, `data/raw` grande)  
-  `requirements.txt`: fijar las versiones exactas (`torch==2.4.0`, `transformers==4.44.0`, etc.)

