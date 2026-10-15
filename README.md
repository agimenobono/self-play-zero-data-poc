# PoC: Self-Play Pretraining with Zero Data

Versión mínima y didáctica del paper [arXiv:2609.30063](https://arxiv.org/abs/2609.30063). Dos transformers diminutos, sin ningún dato real de entrenamiento:

- **Generador**: escribe programas Brainf\*ck.
- **Learner**: predice el siguiente byte de lo que emiten esos programas. Nunca ve el programa.
- **Recompensa**: `r = |⟨∇L(y), P ⊙ (θ_pasado − θ_ahora)⟩|`, el progreso de aprendizaje medido en la geometría de AdamW.

## Ficheros

| Fichero | Qué hace |
|---|---|
| `bf.py` | Máquina universal: Brainf\*ck seguro (cinta circular, mod 256, entrada aleatoria, 5 macros). |
| `core.py` | Todas las piezas, separadas para poder testearlas: modelo, pérdida, recompensa, ventana de snapshots, generador y métricas. |
| `selfplay.py` | El bucle. Escribe en `runs/` un `.csv` (métricas), un `.json` (resumen) y un `_archive.txt` (mejores programas). |
| `tests.py` | Valida cada pieza antes de entrenar. |
| `compare.py` | Agrega semillas, dibuja `runs/comparison.png` e imprime una tabla final. |

## Paso 1: validar las piezas (unos segundos)

```bash
pip install torch matplotlib
python bf.py      # ejemplos de la máquina
python tests.py   # deben pasar todos
```

| Test | Qué garantiza |
|---|---|
| `bf_known_programs` | La máquina produce impares, 3ⁿ mod 256, etc., y nunca falla ni se cuelga. |
| `bf_random_input` | `,` convierte un programa en una distribución de salidas. |
| `causal_mask` | El learner no puede ver el futuro. Si fallara, cualquier resultado sería falso. |
| `padding_does_not_count` | Los ceros de relleno no entran en la pérdida, igual que en el paper ("content tokens"). |
| `init_is_uniform_8_bits` | El modelo sin entrenar está en ~8 bits/byte: el punto de partida es honesto. |
| `learner_can_learn_induction` | El learner tiene capacidad para aprender a copiar; si no, la métrica de inducción no diría nada. |
| `reward_equals_directional_derivative` | **La recompensa calculada con autograd coincide con una diferencia finita.** Valida la fórmula. |
| `snapshot_window` | θ_pasado es siempre el snapshot de la ronda ≈ e/2. |
| `classify` | El detector de familias (aritmética, Fibonacci, geométrica…) acierta. |

## Paso 2: experimentos

```bash
for s in 0 1 2; do
  python selfplay.py --mode selfplay --seed $s
  python selfplay.py --mode uniform  --seed $s
done
python compare.py
```

Para una prueba rápida usa `--rounds 300 --eval-every 50`. Con 3 semillas puedes distinguir señal de ruido; con una sola, no.

**Ablaciones de la recompensa** (tabla 5 del paper): `--reward negate | shuffle | signed | laststep`. La predicción del paper es que `negate` queda peor que `uniform` y `shuffle` peor que `align`. Si ninguna ablación cambia nada, la recompensa no está aportando.

## Qué medir y cómo interpretarlo

Cada fila del CSV corresponde al learner **antes** del paso de esa ronda, así que la fila 0 es el modelo aleatorio.

| Métrica | Qué es | Esperado |
|---|---|---|
| `bpb_python_stdlib` | Pérdida zero-shot en código Python de tu instalación, que el learner nunca ve | Baja desde 8. La línea punteada es la entropía unigrama, un "oráculo" que conoce la frecuencia de cada byte. |
| `bpb_induction` | Bloque aleatorio de 16 bytes repetido; solo se puntúan las repeticiones | Es la prueba de aprendizaje in-context más clara. 8 significa que no copia y 0 que copia perfectamente. |
| `icl_<dataset>` | Pérdida del primer cuarto de la ventana menos la del último cuarto | > 0 indica que usa el contexto. En `random_bytes` debe ser ≈ 0. |
| `bpb_random_bytes` | **Control negativo** | Debe quedarse en ≈ 8. Si baja, hay una fuga o un error. |
| `fresh_bpb` | Pérdida del learner en los programas recién generados, antes de entrenar con ellos | Es la "frontera": ni ≈ 0 (trivial) ni ≈ 8 (ruido). |
| `frac_output`, `unique`, `prog_len` | Salud del generador | Si `unique` → 0, el generador ha colapsado. |
| `fam_*` | Fracción de salidas por familia | `selfplay` debería producir más estructura que `uniform`. |
| `kl` | KL(generador ‖ prior) estimada | Sube a medida que el generador se especializa. Si se dispara, aumenta `--beta`. |
| `first_seen` (en el `.json`) | Primera ronda en que aparece cada familia | Es el análogo de la tabla 1 del paper. |

**Criterio de éxito de la PoC:** con varias semillas, `selfplay-align` debe quedar por debajo de `uniform` en `bpb_python_stdlib` y en `bpb_induction`, con los controles intactos. Con modelos de ~100k parámetros la diferencia puede ser pequeña. Lo importante es poder decir si existe y si es mayor que el ruido entre semillas.

## Revisión: qué se corrigió respecto a la primera versión

1. **El relleno contaba en la pérdida.** El learner aprendía a predecir ceros y la recompensa favorecía programas vacíos. Ahora solo cuentan los bytes emitidos, como en el paper.
2. **El baseline era injusto:** `uniform` veía 32 programas por ronda y `selfplay` 64. Ahora ambos ven los mismos.
3. **La ventana de snapshots sobrescribía θ₀** y se desfasaba una ronda. Ahora está en una clase con test propio.
4. **`nn.TransformerEncoder` era opaco** (tiene caminos rápidos internos). Lo sustituye un bloque causal explícito de 15 líneas, con test de causalidad.
5. **No había forma de comprobar la recompensa.** Ahora hay un test frente a diferencias finitas.
6. **Se añaden** controles, una métrica de aprendizaje in-context, diagnósticos del generador, ablaciones y agregación por semillas.

## Correspondencia con el paper

| Paper | Aquí |
|---|---|
| Transformers Llama, <25M parámetros, contexto 4096 | Transformer causal mínimo, ~100k parámetros, contexto 64 |
| 8 instrucciones + 10 macros | 8 instrucciones + 5 macros |
| Recompensa calculada con JVP (forward mode) | Un `autograd.grad` por programa: más lento, pero más legible y validado con un test |
| GRPO + KL + corrección de importancia off-policy | GRPO + KL, solo sobre muestras on-policy |
| Archivo MAP-Elites por nichos | Top-256 por recompensa, con decaimiento 0,97 |
| Ensembles y leyes de escalado | No incluido |

## Siguientes pasos naturales

- **Fig. 3 del paper:** entrenar un learner nuevo con muestras de generadores tempranos frente a tardíos.
- **Apéndice A.1:** partir del learner de self-play y continuar con texto real, comparándolo con un modelo desde cero.

Código existente: el oficial está enlazado desde el [hilo de Michael Y. Li](https://x.com/michaelyli_/status/2103534415033119001), hay una versión de lucidrains en [PyPI](https://pypi.org/project/self-play-pretrain-zero-data/) y checkpoints en [Hugging Face](https://huggingface.co/rihim/icl-selfplay-vs-text-checkpoints).
