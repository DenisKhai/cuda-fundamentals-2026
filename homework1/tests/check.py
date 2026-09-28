"""Functional checks for the compiled solution"""
import pathlib
import subprocess
import sys


def main():
    executable = str(pathlib.Path(sys.argv[1]).resolve())
    cases = [
        ("2 3 1", "-0.500000 -1.000000"),
        ("1 2 1", "-1.000000"),
        ("1 2 3", "imaginary"),
        ("0 2 -4", "2.000000"),
        ("0 -2 -4", "-2.000000"),
        ("0 0 0", "any"),
        ("0 0 7", "incorrect"),
        ("-1 0 1", "-1.000000 1.000000"),
        ("1 0 -4", "2.000000 -2.000000"),
        ("1 0 0", "0.000000"),
        ("0 1 0", "0.000000"),
        ("1 -3 0", "3.000000 0.000000"),
        ("0.000001 0 -0.000001", "1.000000 -1.000000"),
        ("1 -2 1.000001", "imaginary"),
        ("", None), ("1 2", None), ("x 2 3", None),
        ("nan 2 3", None), ("1 inf 3", None),
        ("1 1e30 1", None),
    ]
    for data, expected in cases:
        result = subprocess.run([executable], input=data + "\n", text=True,
                                capture_output=True, timeout=15)
        assert result.returncode == 0, (data, result)
        if expected is None:
            assert result.stdout == "", (data, result)
            assert result.stderr.startswith("ERROR:"), (data, result)
        else:
            assert result.stdout == expected + "\n", (data, result)
            assert result.stderr == "", (data, result)
    print(f"Passed {len(cases)} cases")


if __name__ == "__main__":
    main()
