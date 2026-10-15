"""Máquina universal mínima (Brainf*ck seguro), al estilo del paper (apéndice E).

- Cinta circular de bytes (mod 256): nunca hay errores de memoria.
- Corchetes sin pareja y caracteres desconocidos = no-op: cualquier cadena es válida.
- ',' lee un byte aleatorio -> un programa define una DISTRIBUCIÓN de salidas.
- Límite de pasos y de salida: siempre termina.
"""
import random

END = "F"                       # token de fin de programa
MACROS = {                      # atajos de un carácter (subconjunto de la tabla 4 del paper)
    "Z": "[-]",                 # pone la celda a 0
    "R": "[->+<]",              # mueve x a la derecha
    "L": "[->+++<]",            # suma 3x a la derecha
    "C": "[->+>+<<]",           # copia x a las dos celdas de la derecha
    "G": "[>]",                 # avanza hasta la siguiente celda a 0
}
ALPHABET = "<>+-[].," + "".join(MACROS) + END


def expand(program: str) -> str:
    return "".join(MACROS.get(c, c) for c in program.replace(END, ""))


def run(program: str, rng: random.Random, out_len=64, max_steps=4096, tape_len=64):
    """Ejecuta el programa y devuelve SOLO los bytes emitidos (sin relleno), como mucho out_len."""
    code = expand(program)
    jump, stack = {}, []
    for i, c in enumerate(code):
        if c == "[":
            stack.append(i)
        elif c == "]" and stack:
            j = stack.pop()
            jump[i], jump[j] = j, i

    tape = [0] * tape_len
    ptr = pc = steps = 0
    out = []
    while pc < len(code) and steps < max_steps and len(out) < out_len:
        c = code[pc]
        if c == ">":
            ptr = (ptr + 1) % tape_len
        elif c == "<":
            ptr = (ptr - 1) % tape_len
        elif c == "+":
            tape[ptr] = (tape[ptr] + 1) % 256
        elif c == "-":
            tape[ptr] = (tape[ptr] - 1) % 256
        elif c == ".":
            out.append(tape[ptr])
        elif c == ",":
            tape[ptr] = rng.randrange(256)
        elif c == "[" and pc in jump and tape[ptr] == 0:
            pc = jump[pc]
        elif c == "]" and pc in jump and tape[ptr] != 0:
            pc = jump[pc]
        pc += 1
        steps += 1
    return out


if __name__ == "__main__":
    rng = random.Random(0)
    for p in ["+[.++]", "+[.L>]", ",[.]", ",[.]", "+++[>+.<-]", "]]x[[+.", "+[]"]:
        print(f"{p:12s} -> {run(p, rng, out_len=10)}")
