"""Functional checks for the compiled solution"""
import pathlib
import subprocess
import sys


executable = str(pathlib.Path(sys.argv[1]).resolve())

cases = [
    ("3\n3.0 -1.0 2.0\n", "-1.000000e+00 2.000000e+00 3.000000e+00"),
    ("5\n9.0 8.0 7.0 5.0 3.0\n",
     "3.000000e+00 5.000000e+00 7.000000e+00 8.000000e+00 9.000000e+00"),
    ("4\n2 1 2 0\n", "0.000000e+00 1.000000e+00 2.000000e+00 2.000000e+00"),
]

for data, expected in cases:
    result = subprocess.run([executable], input=data, text=True, capture_output=True)
    assert result.returncode == 0, result
    assert result.stderr == "", result.stderr
    assert result.stdout.split() == expected.split(), result.stdout

print("Passed 3 cases")
