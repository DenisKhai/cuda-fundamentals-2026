"""Повторить измерения и сохранить CSV для experiments.ipynb."""

import csv
import io
import os
import shutil
import subprocess
from pathlib import Path


lab = Path(__file__).resolve().parents[1]
work = lab / "work"
results = lab / "report/results"
configs = [(1, 32), (8, 64), (16, 128), (32, 128), (64, 128), (128, 256), (256, 256), (512, 512), (1024, 512), (1024, 1024)]


def run(args):
    process = subprocess.run([str(arg) for arg in args], capture_output=True, text=True, timeout=3600)
    if process.returncode != 0 or "ERROR:" in process.stderr:
        raise RuntimeError(process.stderr or process.stdout)
    return process.stdout


if __name__ == "__main__":
    work.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(work)
    nvcc = shutil.which("nvcc") or "/usr/local/cuda-12.8/bin/nvcc"
    run(["g++", "-O2", "-std=c++17", "-DCPU_NO_MAIN", "-c", lab / "report/cpu.cpp", "-o", work / "cpu.o"])
    run([nvcc, "-O2", "-std=c++17", lab / "report/benchmark.cu", work / "cpu.o", "-o", work / "benchmark"])
    for size in [32, 1024, 10000]:
        print(f"Размер {size}×{size}: 100 повторений CPU и каждой из 10 конфигураций GPU", flush=True)
        text = run([work / "benchmark", size, size])
        rows = list(csv.DictReader(io.StringIO(text)))
        assert [(int(row["blocks"]), int(row["threads"])) for row in rows] == configs
        assert all(int(row["repeats"]) == 100 and int(row["cpu_checks"]) == 100 and int(row["gpu_checks"]) == 100 and int(row["errors"]) == 0 for row in rows)
        path = results / f"benchmark-{size}.csv"
        path.write_text(text)
        print(f"Сохранено: {path}", flush=True)
