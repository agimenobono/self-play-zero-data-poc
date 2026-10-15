"""Validación de cada pieza antes de entrenar. Tarda unos segundos en CPU.

  python tests.py        (o pytest tests.py)
"""
import random

import torch

from bf import run
from core import (BOS_LRN, IND_K, SnapshotWindow, TinyLM, classify, ce_matrix, clone, evaluate, execute,
                  grad_align_rewards, log_prior, make_eval_sets, mutate, program_logprob, reward_direction,
                  sample_programs, sample_uniform, seq_loss, token_loss, A, END_ID)


# ---------- máquina ----------
def test_bf_known_programs():
    rng = random.Random(0)
    assert run("+[.++]", rng, out_len=5) == [1, 3, 5, 7, 9]                 # impares
    assert run("+[.L>]", rng, out_len=7) == [1, 3, 9, 27, 81, 243, 217]     # 3^n mod 256
    assert run("+++[>+.<-]", rng) == [1, 2, 3]                               # termina solo
    assert run("]]x[[+.", rng) == [1]                                        # basura -> no falla
    assert run("+[]", rng) == []                                             # bucle infinito -> límite de pasos


def test_bf_random_input_is_a_distribution():
    outs = [run(",[.]", random.Random(s), out_len=6) for s in range(20)]
    assert all(len(set(o)) == 1 for o in outs if o)        # cada salida es un valor repetido...
    assert len({o[0] for o in outs if o}) > 5              # ...pero el valor cambia entre ejecuciones


# ---------- learner ----------
def test_causal_mask():
    model = TinyLM(257, 256, d=32, layers=2)
    x = torch.randint(0, 256, (2, 20))
    y = x.clone()
    y[:, 10:] = torch.randint(0, 256, (2, 10))
    with torch.no_grad():
        assert torch.allclose(model(x)[:, :10], model(y)[:, :10], atol=1e-5)   # el futuro no influye


def test_padding_does_not_count():
    model = TinyLM(257, 256, d=32, layers=1)
    ys = torch.tensor([[1, 3, 5, 7, 0, 0, 0, 0]])
    ys2 = ys.clone()
    ys2[0, 4:] = 99
    lens = torch.tensor([4])
    with torch.no_grad():
        a, b = seq_loss(model, ys, lens), seq_loss(model, ys2, lens)
        c = seq_loss(model, ys[:, :4], lens)
        assert torch.allclose(a, b) and torch.allclose(a, c, atol=1e-6)
        assert seq_loss(model, ys, torch.tensor([0])).item() == 0.0


def test_init_is_uniform_8_bits():
    torch.manual_seed(0)
    res = evaluate(TinyLM(257, 256), make_eval_sets(64, random.Random(0), n=64))
    assert 7.9 < res["bpb_random_bytes"] < 8.15, res


def test_learner_can_learn_and_generalize_induction():
    """Sanidad del learner: entrenado en copias, debe bajar mucho en la tarea de inducción."""
    torch.manual_seed(0)
    rng = random.Random(0)
    model = TinyLM(257, 256, d=64, layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(500):
        rows = []
        for _ in range(32):
            blk = [rng.randrange(1, 256) for _ in range(IND_K)]
            rows.append((blk * 5)[:64])
        ys = torch.tensor(rows)
        loss = token_loss(model, ys, torch.full((32,), 64))
        opt.zero_grad(); loss.backward(); opt.step()
    res = evaluate(model, make_eval_sets(64, random.Random(1), n=64))
    assert res["bpb_induction"] < 6.0, res          # partía de ~8; si falla, el learner no aprende a copiar


# ---------- recompensa ----------
def test_reward_equals_directional_derivative():
    """⟨∇L, v⟩ calculado con autograd == (L(θ+hv) − L(θ−hv)) / 2h. Valida la implementación."""
    torch.manual_seed(0)
    rng = random.Random(0)
    model = TinyLM(257, 256, d=32, layers=1).double()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    params = list(model.parameters())
    ys, lens, _ = execute(["+[.++]", "+[.L>]", ",[.]", "+++[>+.<-]", "+++"], rng, 32)
    past = clone(params)
    for _ in range(3):
        loss = token_loss(model, ys, lens)
        opt.zero_grad(); loss.backward(); opt.step()
    v = reward_direction(opt, params, past)
    r = grad_align_rewards(model, ys, lens, v)
    h = 1e-4
    with torch.no_grad():
        for p_, vi in zip(params, v): p_.add_(h * vi)
        lp = seq_loss(model, ys, lens)
        for p_, vi in zip(params, v): p_.sub_(2 * h * vi)
        lm = seq_loss(model, ys, lens)
        for p_, vi in zip(params, v): p_.add_(h * vi)
    fd = (lp - lm) / (2 * h)
    assert torch.allclose(r.double(), fd, rtol=1e-4, atol=1e-9), (r, fd)
    assert r[-1] == 0.0                                    # "+++" no emite nada -> recompensa 0
    assert r[:-1].abs().min() > 0


def test_zero_direction_gives_zero_reward():
    model = TinyLM(257, 256, d=32, layers=1)
    opt = torch.optim.AdamW(model.parameters())
    params = list(model.parameters())
    ys, lens, _ = execute(["+[.++]"], random.Random(0), 16)
    r = grad_align_rewards(model, ys, lens, reward_direction(opt, params, clone(params)))
    assert r.abs().max() == 0


def test_snapshot_window():
    model = TinyLM(257, 256, d=16, layers=1)
    params = list(model.parameters())
    w = SnapshotWindow(params, every=10)
    for e in range(500):
        k, _ = w.past(e)
        assert k <= e // 2 and e // 2 - k < 10, (e, k)
        w.update(e, params)
    assert len(w.snaps) <= 30                              # no acumula memoria


# ---------- generador ----------
def test_generator_programs_and_logprob():
    torch.manual_seed(0)
    gen = TinyLM(A + 1, A, d=32, layers=1)
    progs = sample_programs(gen, 16, 24)
    assert all(1 <= len(q) <= 24 for q in progs)
    assert all(END_ID not in q[:-1] for q in progs)        # END solo al final
    lp = program_logprob(gen, progs)
    assert (lp <= 0).all()
    # al inicio el generador es casi uniforme: logp ≈ log prior
    assert (lp - log_prior(progs)).abs().mean() < 0.5 * sum(map(len, progs)) / len(progs)
    rng = random.Random(0)
    assert all(len(mutate(q, rng, 24)) <= 24 for q in progs)
    assert all(q[-1] == END_ID or len(q) == 24 for q in sample_uniform(50, 24, rng))


# ---------- medición ----------
def test_classify():
    pre = [9, 9, 9, 9]
    assert classify(pre + [7] * 12) == "constant"
    assert classify(pre + [1, 2, 3] * 5) == "periodic"
    assert classify(pre + list(range(1, 40, 2))) == "arithmetic"
    assert classify(pre + [(i * i) % 256 for i in range(20)]) == "quadratic"
    fib = [1, 1]
    for _ in range(18): fib.append((fib[-1] + fib[-2]) % 256)
    assert classify(pre + fib) == "fibonacci"
    assert classify(pre + [pow(3, i, 256) for i in range(16)]) == "geometric"
    assert classify(pre + [1, 2]) == "short"
    assert classify(pre + [random.Random(0).randrange(256) for _ in range(30)]) == "other"


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    fails = 0
    for name, fn in tests:
        try:
            fn(); print(f"OK    {name}")
        except Exception as ex:
            fails += 1; print(f"FAIL  {name}: {ex!r}")
    print(f"\n{len(tests) - fails}/{len(tests)} tests correctos")
