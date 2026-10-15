"""Piezas del PoC separadas del bucle, para poder testearlas una a una (tests.py)."""
import glob
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from bf import ALPHABET, END, run

A = len(ALPHABET)            # tamaño del alfabeto de programas
END_ID = ALPHABET.index(END)
BOS_GEN = A                  # token de inicio del generador
BOS_LRN = 256                # token de inicio del learner (los bytes son 0..255)
LN2 = math.log(2)
IND_K = 16                   # longitud del bloque en la tarea de inducción
FAMILIES = ["short", "constant", "periodic", "arithmetic", "quadratic", "fibonacci", "geometric", "other"]
STRUCTURED = ["periodic", "arithmetic", "quadratic", "fibonacci", "geometric"]


# ===================== modelo: transformer causal explícito =====================
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # cada posición solo ve el pasado
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    """Decoder-only mínimo. Mismo esqueleto para generador y learner (como en el paper)."""
    def __init__(self, in_vocab, out_vocab, d=64, layers=2, heads=4, ctx=128):
        super().__init__()
        self.tok = nn.Embedding(in_vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, out_vocab)
        nn.init.normal_(self.head.weight, std=0.02)   # al inicio: predicción ~uniforme (≈8 bits/byte)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h))


def clone(params):
    return [p.detach().clone() for p in params]


# ===================== learner =====================
def ce_matrix(model, ys):
    """Entropía cruzada (nats) de predecir cada byte de ys dado lo anterior. (B, T)."""
    x = torch.cat([torch.full_like(ys[:, :1], BOS_LRN), ys[:, :-1]], 1)
    logits = model(x)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ys.reshape(-1), reduction="none")
    return ce.view(ys.shape)


def content_mask(lens, T, dtype):
    return (torch.arange(T)[None] < lens[:, None]).to(dtype)


def seq_loss(model, ys, lens):
    """Pérdida media por secuencia SOLO sobre bytes emitidos (el relleno no cuenta). Vacía -> 0."""
    ce = ce_matrix(model, ys)
    m = content_mask(lens, ys.shape[1], ce.dtype)
    return (ce * m).sum(1) / lens.clamp(min=1).to(ce.dtype)


def token_loss(model, ys, lens):
    """Pérdida media por byte emitido en todo el lote (la que optimiza el learner)."""
    ce = ce_matrix(model, ys)
    m = content_mask(lens, ys.shape[1], ce.dtype)
    return (ce * m).sum() / m.sum().clamp(min=1)


def execute(prog_strs, rng, T):
    outs = [run(s, rng, out_len=T) for s in prog_strs]
    ys = torch.tensor([o + [0] * (T - len(o)) for o in outs], dtype=torch.long)
    lens = torch.tensor([len(o) for o in outs], dtype=torch.long)
    return ys, lens, outs


# ===================== recompensa =====================
def adam_step_operator(opt, params):
    """P = lr / (sqrt(v_hat) + eps) por parámetro: cuánto movería AdamW cada peso por unidad de gradiente."""
    g = opt.param_groups[0]
    lr, (_, b2), eps = g["lr"], g["betas"], g["eps"]
    P = []
    for p in params:
        st = opt.state.get(p, {})
        if "exp_avg_sq" in st:
            v_hat = st["exp_avg_sq"] / (1 - b2 ** float(st["step"]))
            P.append(lr / (v_hat.sqrt() + eps))
        else:
            P.append(torch.full_like(p, lr))
    return P


def reward_direction(opt, params, past):
    """v = P ⊙ (θ_pasado − θ_ahora): la trayectoria reciente del learner, vista por el optimizador."""
    P = adam_step_operator(opt, params)
    return [Pi * (pp - p.detach()) for Pi, pp, p in zip(P, past, params)]


def grad_align_rewards(model, ys, lens, v):
    """⟨∇L(y_i), v⟩ CON signo, un programa cada vez (el abs/ablación se aplica fuera).
    Es la derivada direccional de L(y_i) en la dirección v; el paper la calcula con JVP."""
    params = list(model.parameters())
    out = []
    for i in range(len(ys)):
        if lens[i] == 0:            # sin salida: no hay nada que aprender
            out.append(0.0)
            continue
        loss = seq_loss(model, ys[i:i + 1], lens[i:i + 1])[0]
        g = torch.autograd.grad(loss, params)
        out.append(sum((gi * vi).sum() for gi, vi in zip(g, v)).item())
    return torch.tensor(out)


class SnapshotWindow:
    """Guarda θ_k (pesos tras k pasos) cada `every` rondas y devuelve el más cercano a ⌊e/2⌋ sin pasarse."""
    def __init__(self, params, every):
        self.every = every
        self.snaps = {0: clone(params)}

    def past(self, e):
        k = max(k for k in self.snaps if k <= e // 2)
        return k, self.snaps[k]

    def update(self, e, params):
        """Llamar tras el paso del learner en la ronda e (los pesos ya son θ_{e+1})."""
        if (e + 1) % self.every == 0:
            self.snaps[e + 1] = clone(params)
        keep = max(k for k in self.snaps if k <= (e + 1) // 2)
        self.snaps = {k: s for k, s in self.snaps.items() if k >= keep}


# ===================== generador =====================
@torch.no_grad()
def sample_programs(gen, n, max_len):
    seq = torch.full((n, 1), BOS_GEN, dtype=torch.long)
    for _ in range(max_len):
        tok = torch.distributions.Categorical(logits=gen(seq)[:, -1]).sample()
        seq = torch.cat([seq, tok[:, None]], 1)
    progs = []
    for row in seq[:, 1:].tolist():
        progs.append(row[: row.index(END_ID) + 1] if END_ID in row else row)
    return progs


def sample_uniform(n, max_len, rng):
    """Prior tipo Solomonoff: tokens i.i.d. uniformes hasta END (favorece programas cortos)."""
    progs = []
    for _ in range(n):
        ids = []
        while len(ids) < max_len:
            ids.append(rng.randrange(A))
            if ids[-1] == END_ID:
                break
        progs.append(ids)
    return progs


def program_logprob(gen, progs):
    L = max(len(q) for q in progs)
    ids = torch.tensor([q + [END_ID] * (L - len(q)) for q in progs])
    mask = (torch.arange(L)[None] < torch.tensor([len(q) for q in progs])[:, None]).float()
    x = torch.cat([torch.full((len(progs), 1), BOS_GEN), ids[:, :-1]], 1)
    logp = F.log_softmax(gen(x), -1).gather(-1, ids[..., None]).squeeze(-1)
    return (logp * mask).sum(1)


def log_prior(progs):
    """log g0(x) = −len(x)·log|A| (prior uniforme)."""
    return -torch.tensor([len(q) for q in progs], dtype=torch.float) * math.log(A)


def mutate(ids, rng, max_len):
    body = [t for t in ids if t != END_ID]
    tokens = [t for t in range(A) if t != END_ID]
    op = rng.choice(["sub", "ins", "del"] if body else ["ins"])
    i = rng.randrange(len(body) + (op == "ins"))
    if op == "sub":
        body[i] = rng.choice(tokens)
    elif op == "ins":
        body.insert(i, rng.choice(tokens))
    else:
        body.pop(i)
    return body[: max_len - 1] + [END_ID]


def to_str(ids):
    return "".join(ALPHABET[t] for t in ids if t != END_ID)


# ===================== medición =====================
def classify(seq, lead=4, min_len=10):
    """Familia de la salida (mod 256), como la tabla 1 del paper. Ignora `lead` bytes iniciales."""
    s = seq[lead:]
    if len(s) < min_len:
        return "short"
    if len(set(s)) == 1:
        return "constant"
    for p in range(2, 9):
        if all(s[i] == s[i - p] for i in range(p, len(s))):
            return "periodic"
    d1 = [(b - a) % 256 for a, b in zip(s, s[1:])]
    if len(set(d1)) == 1:
        return "arithmetic"
    d2 = [(b - a) % 256 for a, b in zip(d1, d1[1:])]
    if len(set(d2)) == 1:
        return "quadratic"
    if all((s[i] - s[i - 1] - s[i - 2]) % 256 == 0 for i in range(2, len(s))):
        return "fibonacci"
    for r in (r for r in range(2, 256) if (r * s[0]) % 256 == s[1]):
        if all((r * s[i - 1]) % 256 == s[i] for i in range(1, len(s))):
            return "geometric"
    return "other"


def unigram_entropy(ys):
    """Bits/byte de un 'oráculo' que solo conoce la frecuencia de cada byte en esos datos."""
    c = torch.bincount(ys.reshape(-1), minlength=256).float()
    p = c[c > 0] / c.sum()
    return float(-(p * p.log2()).sum())


def make_eval_sets(T, rng, n=256, extra_files=()):
    raw = {}
    std = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))[:200]
    if std:
        raw["python_stdlib"] = b"".join(open(f, "rb").read() for f in std)
    for path in extra_files:
        raw[os.path.basename(path).replace("_", "-")] = open(path, "rb").read()
    raw["random_bytes"] = bytes(rng.randrange(256) for _ in range(n * T * 2))   # control
    sets = {}
    for name, data in raw.items():
        starts = [rng.randrange(len(data) - T) for _ in range(n)]
        sets[name] = torch.tensor([list(data[s:s + T]) for s in starts])
    # tarea in-context: un bloque aleatorio de IND_K bytes repetido. Solo se puntúan las repeticiones.
    rows = []
    for _ in range(n):
        block = [rng.randrange(1, 256) for _ in range(IND_K)]
        rows.append((block * (T // IND_K + 1))[:T])
    sets["induction"] = torch.tensor(rows)
    return sets


@torch.no_grad()
def evaluate(model, sets):
    res = {}
    for name, ys in sets.items():
        ce = ce_matrix(model, ys) / LN2                       # bits
        if name == "induction":
            res["bpb_induction"] = ce[:, IND_K:].mean().item()
        else:
            q = ys.shape[1] // 4
            res[f"bpb_{name}"] = ce.mean().item()
            res[f"icl_{name}"] = (ce[:, :q].mean() - ce[:, -q:].mean()).item()   # >0: usa el contexto
    return res
