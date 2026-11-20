import argparse
from pathlib import Path
import random
import struct
import subprocess

parser = argparse.ArgumentParser(description='Проверка бинарной сортировки uint32.')
parser.add_argument('executable')
args = parser.parse_args()
executable = str(Path(args.executable).resolve())
lab = Path(__file__).resolve().parents[1]


def run(data):
    return subprocess.run([executable], input=data, capture_output=True, timeout=120)


def check(values):
    data = struct.pack('<I', len(values)) + struct.pack(f'<{len(values)}I', *values)
    process = run(data)
    assert process.returncode == 0 and not process.stderr, process.stderr.decode(errors='replace')
    expected = struct.pack(f'<{len(values)}I', *sorted(values))
    assert process.stdout == expected, f'Неверная сортировка для n={len(values)}'


cases = [[], [0], [0xffffffff], [0xffffffff, 0, 0x80000000, 0x7fffffff],
         list(range(10)), list(range(9, -1, -1)), [7] * 1025,
         [1 << i for i in range(31, -1, -1)], [0, 0xffffffff] * 513]
rng = random.Random(508)
for n in [31, 32, 33, 63, 64, 65, 511, 512, 513, 1023, 1024, 1025, 32767, 262145]:
    cases.append([rng.getrandbits(32) for _ in range(n)])
for values in cases:
    check(values)

example = (lab / 'data/int10').read_bytes()
process = run(example)
assert process.returncode == 0 and not process.stderr
assert process.stdout == (lab / 'data/uint10.sorted').read_bytes()
print(f'Пройдено тестов: {len(cases) + 1} (включая пример из ZIP).')

invalid = [b'', b'\x01', struct.pack('<I', 128000001), struct.pack('<i', -1),
           struct.pack('<I', 2) + struct.pack('<I', 3), struct.pack('<I', 1) + b'\x00']
for data in invalid:
    process = run(data)
    assert process.returncode == 0 and not process.stdout and process.stderr.startswith(b'ERROR:'), data
print(f'Проверок ошибочного ввода: {len(invalid)}.')
print('Все тесты пройдены успешно.')
