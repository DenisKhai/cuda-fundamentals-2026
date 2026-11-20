# Лабораторная работа №5, вариант 8

Поразрядная сортировка `uint32` на CUDA. Собственное рекурсивное сканирование Blelloch с разделяемой памятью.

```bash
cd lab5-variant8
nvcc -O2 -std=c++17 lab5.cu -o lab5
mkdir -p work
./lab5 < data/int10 > work/result.bin
python3 tests/check.py ./lab5
```

Вход: бинарные `uint32 n` и `n` чисел. Выход: только отсортированные числа, без размера. Максимум — 128 000 000 элементов. При ошибке — `ERROR:` в stderr и код возврата 0.

Замеры, проверки и выводы — в [experiments.ipynb](experiments.ipynb). Запустить измерение времени работы CPU и GPU и профилирование ядер CUDA:

```bash
python3 -B report/run_benchmarks.py --profile
```
