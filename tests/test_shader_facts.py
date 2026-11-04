#!/usr/bin/env python3
"""Portable embedder guard; no Metal SDK, dependencies or source-tree edits.
Run: python3 tests/test_shader_facts.py (honors CC).
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FACTS = {
    "LUNELETTE_MODEL_NORM_EPS": ("1e-5f", "2e-4f"),
    "LUNELETTE_MODEL_QUERY_SCALE": ("0.1f", "0.25f"),
    "LUNELETTE_MODEL_ORIGINAL_CONTEXT": ("16384", "8192"),
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


with tempfile.TemporaryDirectory(prefix="lunelette-shader-facts-") as directory:
    temp = Path(directory)
    (temp / "tools").mkdir()
    (temp / "include").mkdir()
    source = ROOT / "tools/embed_shader.c"
    (temp / "tools/embed_shader.c").write_bytes(source.read_bytes())
    header = (ROOT / "include/model_facts.h").read_text()
    shader = ROOT / "src/metal_kernels.metal"
    for altered in (False, True):
        facts = header
        for name, (original, replacement) in FACTS.items():
            expected = f"#define {name} {original}"
            check(expected in facts, f"missing independent baseline: {expected}")
            if altered:
                facts = facts.replace(expected, f"#define {name} {replacement}")
        (temp / "include/model_facts.h").write_text(facts)
        executable = temp / "embed_shader"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-Wall", "-Wextra", "-Wpedantic",
            str(temp / "tools/embed_shader.c"), "-o", str(executable)], check=True)
        output = subprocess.check_output([str(executable), str(shader)], text=True)
        initializer = output.split("{", 1)[1].split("}", 1)[0]
        data = bytes(int(value) for value in re.findall(r"\d+", initializer))
        check(data[-1:] == b"\0", "shader must be NUL terminated")
        text = data[:-1].decode()
        prefix, body = text.split("#include", 1)
        for name, values in FACTS.items():
            check(f"#define {name} {values[int(altered)]}\n" in prefix,
                  f"embedded fact did not track header: {name}")
            check(body.count(name) == 1, f"shader must consume {name}")
        check("+ LUNELETTE_MODEL_NORM_EPS" in body, "norm fact not used")
        check("1.0f + LUNELETTE_MODEL_QUERY_SCALE *" in body, "query fact not used")
        check("positions[token] / LUNELETTE_MODEL_ORIGINAL_CONTEXT" in body,
              "context division must retain integer semantics")
print("shader facts: original and isolated altered headers passed")
