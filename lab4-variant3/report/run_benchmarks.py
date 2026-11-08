import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

LAB = Path(__file__).resolve().parents[1]
WORK = LAB / 'work'
RESULTS = LAB / 'report/results'


def run(command):
    process = subprocess.run(list(map(str, command)), cwd=LAB, capture_output=True, text=True, timeout=7200)
    if process.returncode or 'ERROR:' in process.stderr:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or f'Ошибка запуска {command[0]}')
    return process.stdout


def validate_csv(text, n, args):
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == (1 if args.cpu_only else 10), 'Неполные результаты'
    for row in rows:
        assert row['mode'] == ('cpu' if args.cpu_only else 'gpu') and int(row['n']) == n
        assert int(row['repeats']) == args.repeats and int(row['warmup']) == args.warmup
        assert int(row['cpu_checks']) == args.repeats + args.warmup and int(row['errors']) == 0
        assert int(row['gpu_checks']) == (0 if args.cpu_only else args.repeats + args.warmup)
        for prefix in ['cpu'] if args.cpu_only else ['cpu', 'gpu']:
            ms, error, residual = [float(row[f'{prefix}_{key}']) for key in ['ms', 'rel_error', 'scaled_residual']]
            assert math.isfinite(ms) and ms > 0
            assert math.isfinite(error) and 0 <= error <= 1e-10
            assert math.isfinite(residual) and 0 <= residual <= 1e-10
        if not args.cpu_only:
            assert math.isclose(float(row['speedup']), float(row['cpu_ms']) / float(row['gpu_ms']), rel_tol=1e-9)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--sizes', type=int, nargs='+', default=[32, 256, 1024])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-size', type=int, default=256)
    parser.add_argument('--nvcc', default=shutil.which('nvcc') or '/usr/local/cuda-12.8/bin/nvcc')
    parser.add_argument('--arch', default='sm_61')
    args = parser.parse_args()
    if (any(not 1 <= n <= 10000 for n in [*args.sizes, args.profile_size])
            or not 1 <= args.repeats <= 10000 or not 0 <= args.warmup <= 10000
            or len(set(args.sizes)) != len(args.sizes) or (args.cpu_only and args.profile)):
        parser.error('Некорректные размеры, повторы или режим профилирования')
    for name, folder in {'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp', 'CUDA_CACHE_PATH': 'cuda-cache',
                         'NVPROF_TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache'}.items():
        path = WORK / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    RESULTS.mkdir(parents=True, exist_ok=True)
    mode = 'cpu' if args.cpu_only else 'gpu'
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), status='running', mode=mode,
                    measurements_status='not_started', arguments=vars(args), result_files=[],
                    platform=platform.platform(), cpu_model=platform.processor())
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            metadata['cpu_model'] = line.split(':', 1)[1].strip()
            break
    log = RESULTS / f'nvprof-{args.profile_size}.log'
    if args.profile:
        metadata['profile_status'] = 'not_started'
        log.write_text('Profiler has not been started.\n')
    try:
        metadata['cxx_version'] = run(['g++', '--version']).splitlines()[0]
        flags = ['-O2', '-std=c++17', '-DCPU_NO_MAIN']
        sources = [LAB / 'report/benchmark.cu', LAB / 'report/cpu.cpp']
        binary = WORK / ('benchmark-cpu' if args.cpu_only else 'benchmark')
        if args.cpu_only:
            run(['g++', *flags, '-DBENCHMARK_CPU_ONLY', '-x', 'c++', *sources, '-o', binary])
        else:
            metadata['nvcc_version'] = run([args.nvcc, '--version'])
            run([args.nvcc, *flags, f'-arch={args.arch}', *sources, '-o', binary])
            run([args.nvcc, *flags, f'-arch={args.arch}', LAB / 'report/device_info.cu', '-o', WORK / 'device_info'])
            metadata['device_info'] = run([WORK / 'device_info'])
            (RESULTS / 'device-info.txt').write_text(metadata['device_info'])
        metadata['measurements_status'] = 'running'
        for n in args.sizes:
            print(f'n={n}: {args.repeats} измерений, прогрев {args.warmup}', flush=True)
            text = run([binary, n, args.repeats, args.warmup])
            validate_csv(text, n, args)
            path = RESULTS / f'{"cpu" if args.cpu_only else "benchmark"}-{n}.csv'
            path.write_text(text)
            metadata['result_files'].append(path.name)
        metadata['measurements_status'] = 'complete'
        if args.profile:
            nvprof = shutil.which('nvprof') or str(Path(args.nvcc).resolve().with_name('nvprof'))
            metadata['profile_status'] = 'running'
            log.write_text('')
            output = run([nvprof, '--profile-from-start', 'off', '--log-file', log,
                          binary, args.profile_size, 1, args.warmup, '--profile'])
            profile = log.read_text()
            assert 'Profiled solver passed:' in output and 'back_substitute' in profile and 'error:' not in profile.lower()
            metadata.update(profile_status='complete', profile_log=log.name, profile_output=output.strip())
        metadata['status'] = 'complete'
        (RESULTS / f'run-error-{mode}.log').unlink(missing_ok=True)
        return 0
    except Exception as error:
        metadata.update(status='failed', error=str(error) or 'Проверка результатов не пройдена')
        message = 'ERROR: ' + metadata['error'] + '\n'
        (RESULTS / f'run-error-{mode}.log').write_text(message)
        if args.profile:
            if metadata['profile_status'] == 'running':
                metadata['profile_status'] = 'failed'
            log.write_text(log.read_text() + '\n' + message)
        print(message, file=sys.stderr, end='')
        return 1
    finally:
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (RESULTS / f'metadata-{mode}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
