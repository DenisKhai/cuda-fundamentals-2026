import argparse
import math
from pathlib import Path
import random
import subprocess


def make_case(matrix, solution):
    """Вычисляем правую часть системы по заданной матрице и известному решению."""
    rhs = [math.fsum(a * x for a, x in zip(row, solution)) for row in matrix]
    return matrix, rhs, solution


def cases():
    examples = [
        ([[1, 2], [3, 4]], [-4, 4.5]),
        ([[-2]], [3.5]), ([[1]], [1.00000000012345]),
        ([[1e-200]], [-2.5]), ([[1e200]], [0.125]),
        ([[2, 1], [1, 3]], [0, 0]),
        ([[int(i == j) for j in range(5)] for i in range(5)], [0, -3.25, 0, 7.5, 1]),
        ([[1, 2], [-8, 3]], [-3, 1.25]),
        ([[1e-100, 1], [1, 1]], [2, -1]),
        ([[0, 0, 2], [3, 0, 1], [1, -4, 2]], [2, -3, 0.5]),
        ([[1e-12, 0, 0], [0, -2e-20, 0], [0, 0, 4e-30]], [-2, 0.25, 3]),
        ([[2, 0, 0], [0, -4, 0], [0, 0, 8]], [1e-12, -1e12, 0.125]),
    ]
    for matrix, solution in examples:
        yield make_case(matrix, solution)
    rng = random.Random(403)
    for n in (3, 7, 31, 32, 33, 65, 129, 257, 1031):
        matrix = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
        for i, row in enumerate(matrix):
            row[i] = (1 if i % 2 else -1) * (1 + sum(abs(v) for j, v in enumerate(row) if j != i))
        rng.shuffle(matrix)
        solution = [rng.choice(list(range(-32, 0)) + list(range(1, 33))) / 8 for _ in range(n)]
        yield make_case(matrix, solution)


def run(data):
    return subprocess.run([executable], input=data, text=True, capture_output=True, timeout=60)


def check(matrix, rhs, expected):
    lines = [str(len(matrix))] + [' '.join(map(str, row)) for row in matrix]
    result = run('\n'.join(lines + [' '.join(map(str, rhs))]) + '\n')
    assert result.returncode == 0 and not result.stderr, result.stderr
    actual = list(map(float, result.stdout.split()))
    assert len(actual) == len(expected), 'Неверное число элементов решения'
    for value, reference in zip(actual, expected):
        assert math.isfinite(value) and math.isclose(
            value, reference, rel_tol=1e-10, abs_tol=1e-12 if reference == 0 else 0
        ), f'Получено {value}, ожидалось {reference}'
    for row, b in zip(matrix, rhs):
        products = [a * x for a, x in zip(row, actual)]
        scale = math.fsum(abs(v) for v in products) + abs(b)
        assert abs(math.fsum(products) - b) <= 2e-10 * scale, 'Слишком большая невязка'


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('executable', help='Путь к программе CPU или GPU')
parser.add_argument('--invalid', action='store_true', help='Проверить ошибочный ввод')
args = parser.parse_args()
executable = str(Path(args.executable).resolve())
for number, case in enumerate(cases(), 1):
    check(*case)
print(f'Пройдено тестов: {number}')

if args.invalid:
    inputs = ['', 'size\n', '0\n', '-1\n', '10001\n', '2\n1 2\n3\n',
              '2\n1 0\n0 1\n1\n', '1\nnan\n1\n', '1\n1\ninf\n',
              '2\n1 2\n2 4\n3 6\n', '1\n0\n0\n', '1\n0\n1\n']
    for data in inputs:
        result = run(data)
        assert result.returncode == 1 and not result.stdout and result.stderr.startswith('ERROR:'), data
    print(f'Проверок ошибочного ввода: {len(inputs)}')

print('Все тесты пройдены успешно.')
