"""Bucle de self-play (PoC de arXiv:2609.30063). Ver README para los experimentos.

  python selfplay.py --mode selfplay                 # método completo
  python selfplay.py --mode uniform                  # baseline: programas del prior, sin generador
  python selfplay.py --mode selfplay --reward negate # ablación de la recompensa
"""
import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter

import torch

from core import (A, FAMILIES, LN2, STRUCTURED, SnapshotWindow, TinyLM, classify, clone, evaluate,
                  execute, grad_align_rewards, log_prior, make_eval_sets, mutate, program_logprob,
                  reward_direction, sample_programs, sample_uniform, seq_loss, to_str, token_loss,
                  unigram_entropy)

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["selfplay", "uniform"], default="selfplay")
p.add_argument("--reward", choices=["align", "signed", "negate", "shuffle", "laststep"], default="align",
               help="ablaciones de la tabla 5 del paper (solo con --mode selfplay)")
p.add_argument("--rounds", type=int, default=3000)
p.add_argument("--fresh", type=int, default=32)        # programas nuevos por ronda
p.add_argument("--replay", type=int, default=16)       # reproducidos del archivo
p.add_argument("--mutate", type=int, default=16)       # mutaciones del archivo
p.add_argument("--max-prog-len", type=int, default=24)
p.add_argument("--seq-len", type=int, default=64)      # bytes de salida = contexto del learner
p.add_argument("--d-model", type=int, default=64)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--lr-learner", type=float, default=1e-3)
p.add_argument("--lr-gen", type=float, default=3e-4)
p.add_argument("--beta", type=float, default=0.02)     # KL hacia el prior uniforme
p.add_argument("--ei-coef", type=float, default=1.0)   # peso del SFT ponderado (expert iteration)
p.add_argument("--snap-every", type=int, default=10)
p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--eval-files", nargs="*", default=[])  # tus propios ficheros 'naturales'
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="runs")
args = p.parse_args()

selfplay = args.mode == "selfplay"
tag = (f"selfplay-{args.reward}" if selfplay else "uniform") + f"_s{args.seed}"
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
torch.set_num_threads(os.cpu_count() or 1)

learner = TinyLM(257, 256, args.d_model, args.layers)
generator = TinyLM(A + 1, A, args.d_model, args.layers)
opt_l = torch.optim.AdamW(learner.parameters(), lr=args.lr_learner)
opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr_gen)
lparams = list(learner.parameters())
window = SnapshotWindow(lparams, args.snap_every)
prev = clone(lparams)                                   # θ_{e-1}, solo para la ablación 'laststep'

eval_sets = make_eval_sets(args.seq_len, random.Random(1234), extra_files=args.eval_files)
refs = {f"unigram_{k}": unigram_entropy(v) for k, v in eval_sets.items() if k != "induction"}
n_total = args.fresh + args.replay + args.mutate        # mismo nº de programas por ronda en ambos modos

archive, first_seen = {}, {}
fam_total, fam_window = Counter(), Counter()
f_log = open(os.path.join(args.out, f"{tag}.csv"), "w", newline="")
writer, t0, tokens = None, time.time(), 0
print(f"== {tag} | learner {sum(q.numel() for q in lparams):,} parámetros | refs {refs}")

for e in range(args.rounds + 1):
    # ---- 1) pool de programas ----
    if selfplay:
        fresh = sample_programs(generator, args.fresh if archive else n_total, args.max_prog_len)
        elites = list(archive)
        pool = ([list(rng.choice(elites)) for _ in range(args.replay)] +
                [mutate(list(rng.choice(elites)), rng, args.max_prog_len) for _ in range(args.mutate)]
                ) if elites else []
    else:
        fresh, pool = sample_uniform(n_total, args.max_prog_len, rng), []
    progs = fresh + pool
    n = len(fresh)
    strs = [to_str(q) for q in progs]

    # ---- 2) ejecutar: el learner solo verá estas salidas, nunca los programas ----
    ys, lens, outs = execute(strs, rng, args.seq_len)
    for i, (s, o) in enumerate(zip(strs, outs)):
        fam = classify(o)
        if i < n:
            fam_window[fam] += 1
            fam_total[fam] += 1
        if fam in STRUCTURED and fam not in first_seen:
            first_seen[fam] = {"round": e, "program": s, "output": o[:16]}

    # ---- 3) recompensa, con el learner ANTES de actualizarlo ----
    if selfplay:
        past = prev if args.reward == "laststep" else window.past(e)[1]
        r_raw = grad_align_rewards(learner, ys, lens, reward_direction(opt_l, lparams, past))
        r = {"align": r_raw.abs(), "laststep": r_raw.abs(), "signed": r_raw, "negate": -r_raw.abs(),
             "shuffle": r_raw.abs()[torch.randperm(len(r_raw))]}[args.reward]
        logp = program_logprob(generator, progs)
        logp0 = log_prior(progs)

    # ---- 4) medir (θ_e, antes del paso de esta ronda: la fila 0 es el modelo aleatorio) ----
    if e % args.eval_every == 0:
        with torch.no_grad():
            m = lens[:n] > 0
            fresh_bpb = (seq_loss(learner, ys[:n][m], lens[:n][m]).mean() / LN2).item() if m.any() else math.nan
        tot = max(sum(fam_window.values()), 1)
        row = {"round": e, "tokens": tokens, "seconds": round(time.time() - t0, 1),
               "fresh_bpb": fresh_bpb,                                    # dificultad de los datos nuevos
               "frac_output": m.float().mean().item(),                   # programas que emiten algo
               "unique": len(set(strs[:n])) / n,                         # diversidad
               "prog_len": sum(len(s) for s in strs[:n]) / n,
               "out_len": lens[:n].float().mean().item(),
               "kl": (logp[:n] - logp0[:n]).mean().item() if selfplay else 0.0,
               "r_mean": r[:n].mean().item() if selfplay else math.nan,
               "r_std": r[:n].std().item() if selfplay else math.nan,
               "archive": len(archive),
               **{f"fam_{f}": fam_window[f] / tot for f in FAMILIES},
               **evaluate(learner, eval_sets)}
        fam_window.clear()
        if writer is None:
            writer = csv.DictWriter(f_log, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        f_log.flush()
        structured = sum(row[f"fam_{f}"] for f in STRUCTURED)
        print(f"[{tag}] r{e:5d} {row['seconds']:6.0f}s | python {row.get('bpb_python_stdlib', math.nan):.2f} "
              f"| induc {row['bpb_induction']:.2f} | random {row['bpb_random_bytes']:.2f} "
              f"| fresh {fresh_bpb:.2f} | salida {row['frac_output']:.2f} | estruct {structured:.2f} | kl {row['kl']:.2f}")
        for ids, rq in list(archive.items())[:3]:
            s = to_str(ids)
            print(f"      r={rq:.2e}  {s:24s} -> {execute([s], random.Random(0), 12)[2][0]}")

    # ---- 5) learner: un paso de next-byte sobre los bytes emitidos ----
    if args.reward == "laststep":
        prev = clone(lparams)
    loss = token_loss(learner, ys, lens)
    opt_l.zero_grad()
    loss.backward()
    opt_l.step()
    tokens += int(lens.sum())
    window.update(e, lparams)

    # ---- 6) generador: policy gradient (GRPO) + KL al prior + SFT ponderado ----
    if selfplay:
        adv = (r - r.mean()) / (r.std() + 1e-8) - args.beta * (logp.detach() - logp0)
        pg = -(adv[:n] * logp[:n]).mean()                  # solo muestras on-policy
        w = r.clamp(min=0)
        w = w / (w.sum() + 1e-12)
        ei = -(w * logp).sum()
        opt_g.zero_grad()
        (pg + args.ei_coef * ei).backward()
        opt_g.step()

        for k in archive:                                  # las recompensas antiguas caducan
            archive[k] *= 0.97
        for q, rq in zip(progs, r.tolist()):
            if rq > 0:
                archive[tuple(q)] = max(rq, archive.get(tuple(q), 0.0))
        archive = dict(sorted(archive.items(), key=lambda kv: -kv[1])[:256])

f_log.close()
summary = {"args": vars(args), "refs": refs, "first_seen": first_seen,
           "families_total": dict(fam_total), "final": row}
json.dump(summary, open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)
with open(os.path.join(args.out, f"{tag}_archive.txt"), "w") as f:
    for ids, rq in archive.items():
        s = to_str(ids)
        o = execute([s], random.Random(0), 16)[2][0]
        f.write(f"{rq:.3e}\t{classify(o, lead=0, min_len=8)}\t{s}\t{o}\n")
print(f"Guardado en {args.out}/{tag}.*  | primeras apariciones: {first_seen}")
