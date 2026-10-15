"""Agrega las ejecuciones de runs/ (media y rango entre semillas) y dibuja las métricas clave.

  python compare.py            # lee runs/, escribe runs/comparison.png
"""
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
STRUCT = ["fam_periodic", "fam_arithmetic", "fam_quadratic", "fam_fibonacci", "fam_geometric"]
PANELS = [  # (métrica, título, línea de referencia)
    ("bpb_python_stdlib", "Zero-shot en Python (bits/byte)", "unigram_python_stdlib"),
    ("bpb_induction", "Inducción: repetir lo visto (bits/byte)", 8.0),
    ("icl_python_stdlib", "Ganancia in-context en Python (bits)", 0.0),
    ("bpb_random_bytes", "Control: bytes aleatorios (debe ≈ 8)", 8.0),
    ("fresh_bpb", "Dificultad de los programas nuevos", None),
    ("structured", "Fracción de salidas con estructura", None),
    ("frac_output", "Fracción de programas que emiten algo", None),
    ("kl", "KL(generador ‖ prior uniforme)", None),
]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


groups, summaries = defaultdict(list), {}
for path in sorted(glob.glob(os.path.join(RUNS, "*.csv"))):
    tag = os.path.basename(path)[:-4]
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["structured"] = sum(num(r[k]) for k in STRUCT)
    groups[tag.rsplit("_s", 1)[0]].append(rows)
    if os.path.exists(path[:-4] + ".json"):
        summaries[tag] = json.load(open(path[:-4] + ".json"))
if not groups:
    sys.exit(f"No hay ficheros .csv en {RUNS}/")
refs = next(iter(summaries.values()))["refs"] if summaries else {}

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, (metric, title, ref) in zip(axes.flat, PANELS):
    for label, runs in sorted(groups.items()):
        if metric not in runs[0][0]:
            continue
        L = min(len(r) for r in runs)
        x = [num(runs[0][i]["round"]) for i in range(L)]
        vals = [[num(r[i][metric]) for r in runs] for i in range(L)]
        mean = [sum(v) / len(v) for v in vals]
        line, = ax.plot(x, mean, label=f"{label} (n={len(runs)})")
        if len(runs) > 1:
            ax.fill_between(x, [min(v) for v in vals], [max(v) for v in vals], alpha=0.2, color=line.get_color())
    ref_val = refs.get(ref) if isinstance(ref, str) else ref
    if ref_val is not None:
        ax.axhline(ref_val, color="gray", ls=":", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("ronda")
axes.flat[0].legend(fontsize=8)
plt.tight_layout()
out = os.path.join(RUNS, "comparison.png")
plt.savefig(out, dpi=110)

print(f"\n{'configuración':24s} {'seeds':>5s} {'python':>8s} {'inducción':>10s} {'icl':>6s} {'random':>7s}")
for label, runs in sorted(groups.items()):
    last = [r[-1] for r in runs]
    f = lambda k: sum(num(r.get(k)) for r in last) / len(last)
    print(f"{label:24s} {len(runs):5d} {f('bpb_python_stdlib'):8.2f} {f('bpb_induction'):10.2f} "
          f"{f('icl_python_stdlib'):6.2f} {f('bpb_random_bytes'):7.2f}")
print(f"\nReferencia: entropía unigrama de Python = {refs.get('unigram_python_stdlib', math.nan):.2f} bits/byte")
print("\nPrimera aparición de cada familia (ronda: programa -> salida):")
for tag, s in sorted(summaries.items()):
    fs = s.get("first_seen", {})
    print(f"  {tag}: " + ("; ".join(f"{k} r{v['round']}: {v['program']} -> {v['output'][:6]}" for k, v in fs.items()) or "ninguna"))
print(f"\nGuardado: {out}")
