"""Functional checks for the compiled solution"""
import math
import pathlib
import re
import subprocess
import sys


executable = str(pathlib.Path(sys.argv[1]).resolve())

cases = [
    ([1, 5, 3], [4, 2, 6]),
    ([-4, 2.5, -1], [3, -2, -1]),
    ([1.0000000002], [1.0000000001]),
    ([0, 2, -3], [0, 2, -3]),
    ([i % 100 for i in range(65539)],
     [-1 - i % 100 for i in range(65539)]),
]

for number, (a, b) in enumerate(cases, 1):
    data = str(len(a)) + "\n"
    data += " ".join(map(str, a)) + "\n"
    data += " ".join(map(str, b)) + "\n"

    result = subprocess.run(
        [executable], input=data, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result
    assert result.stderr == "", result.stderr

    values = result.stdout.split()
    assert len(values) == len(a), f"Case {number}: wrong number of elements"

    for i, value in enumerate(values):
        assert re.fullmatch(r"-?\d\.\d{10}e[+-]\d{2,3}", value), value
        expected = min(a[i], b[i])
        actual = float(value)
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=0), (
            f"Case {number}, element {i}: {actual} instead of {expected}"
        )

print(f"Passed {len(cases)} cases")
