# Лабораторная работа №4, вариант 3

Решение квадратной СЛАУ методом Гаусса с выбором главного элемента по столбцу.

## Компиляция CUDA-кода

```bash
cd lab4-variant3
nvcc -O2 -std=c++17 lab4.cu -o lab4
```

## Запуск программы

```bash
./lab4 < data/example.txt
```

На вход подаются размер `n`, матрица `n × n` и вектор `b`. На выходе — вектор решения `x`.

## Запуск тестов

```bash
python3 tests/check.py ./lab4 --invalid
```

## Jupyter Notebook

В [`experiments.ipynb`](experiments.ipynb) представлены метод, характеристики окружения, эксперименты CPU/GPU и профилирование `nvprof`. Сохранённые результаты читаются из `report/results/`.

Повторить измерения и профилирование:

```bash
python3 report/run_benchmarks.py --profile
```
