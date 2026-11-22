"""Сборка лабораторной №6 на текущем компьютере с GTX 1050."""

import os
from pathlib import Path
import subprocess

lab = Path(__file__).resolve().parent
work = lab / "work"
work.mkdir(exist_ok=True)

# Временные файлы и кэши остаются внутри проекта.
for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
    os.environ[name] = str(work)
os.environ["CUDA_CACHE_DISABLE"] = "1"

# Компилятор CUDA, исходник и параметры сборки.
subprocess.run([
    "/usr/local/cuda/bin/nvcc",
    "-O2", "-std=c++14", "-arch=sm_61",
    "-Werror", "cross-execution-space-call",
    "-Iinclude", "lab6.cu", "-o", "lab6",
    "-lm", "-lcublas", "-lcurand",
    # Имена установленных в системе библиотек OpenGL.
    "-l:libGL.so.1", "-l:libGLU.so.1",
    "-l:libglut.so.3", "-l:libGLEW.so.2.2",
], cwd=lab, check=True)

print("Готово:", lab / "lab6")
