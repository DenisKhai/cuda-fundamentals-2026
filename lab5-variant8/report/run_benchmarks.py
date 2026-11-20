import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

LAB = Path(__file__).resolve().parents[1]
WORK = LAB / 'work'
RESULTS = LAB / 'report/results'
CONFIGS = [(1, 32), (32, 128), (128, 256), (256, 256), (1024, 1024)]


def run(command):
    process = subprocess.run(list(map(str, command)), cwd=LAB, capture_output=True, timeout=7200)
    if process.returncode or b'ERROR:' in process.stderr:
        raise RuntimeError(process.stderr.decode(errors='replace')[-1500:] or 'Ошибка запуска программы')
    return process.stdout.decode()


def parse_profile(text):
    result = []
    section = None
    pattern = re.compile(r'^\s*([\d.]+)%\s+([\d.]+)(ns|us|ms|s)\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(.+)$')
    for line in text.splitlines():
        for label in ['GPU activities', 'API calls']:
            if label + ':' in line:
                section = label
                line = line.split(label + ':', 1)[1]
        match = pattern.fullmatch(line)
        if section and match:
            percent, value, unit, calls, name = match.groups()
            result.append(dict(section=section, name=name.split('(')[0], percent=float(percent),
                               ms=float(value) * {'ns': 1e-6, 'us': 1e-3, 'ms': 1, 's': 1000}[unit], calls=int(calls)))
    return result


def main():
    parser = argparse.ArgumentParser(description='Замеры CPU/GPU и времени отдельных ядер nvprof.')
    parser.add_argument('--sizes', nargs='+', type=int, default=[32, 1048576, 128000000])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--warmup-size', type=int, default=4096)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-slow', action='store_true', help='Также профилировать предельный массив одним блоком (очень долго)')
    parser.add_argument('--nvcc', default=shutil.which('nvcc') or '/usr/local/cuda-12.8/bin/nvcc')
    parser.add_argument('--arch', default='sm_61')
    args = parser.parse_args()
    if (any(not 1 <= n <= 128000000 for n in args.sizes) or len(set(args.sizes)) != len(args.sizes)
            or args.repeats < 1 or args.warmup < 0 or args.warmup_size < 1):
        parser.error('Некорректные параметры')
    for name, folder in {'TMPDIR': 'tmp', 'CUDA_CACHE_PATH': 'cuda-cache', 'NVPROF_TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache'}.items():
        path = WORK / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    RESULTS.mkdir(parents=True, exist_ok=True)
    metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), arguments=vars(args).copy(),
                    platform=platform.platform(), result_files=[], profile_files=[])
    metadata['arguments']['nvcc'] = Path(args.nvcc).name
    (RESULTS / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    try:
        flags = ['-O2', '-std=c++17']
        metadata['nvcc_version'] = run([args.nvcc, '--version']).strip()
        metadata['cxx_version'] = run(['g++', '--version']).splitlines()[0]
        cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
        memory = next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))
        metadata.update(cpu_model=cpu, ram_kib=int(memory), logical_cpus=os.cpu_count())
        metadata['disks'] = json.loads(run(['lsblk', '-J', '-b', '-d', '-o', 'NAME,MODEL,SIZE,ROTA,TYPE']))['blockdevices']
        metadata['disks'] = [d for d in metadata['disks'] if d['type'] == 'disk']
        metadata['gpu_driver'] = run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']).strip()
        run([args.nvcc, *flags, f'-arch={args.arch}', LAB / 'lab5.cu', '-o', WORK / 'lab5'])
        run(['g++', *flags, LAB / 'report/cpu.cpp', '-o', WORK / 'cpu'])
        run([args.nvcc, *flags, f'-arch={args.arch}', '-DCPU_NO_MAIN', LAB / 'report/benchmark.cu', LAB / 'report/cpu.cpp', '-o', WORK / 'benchmark'])
        run([args.nvcc, *flags, f'-arch={args.arch}', LAB / 'report/device_info.cu', '-o', WORK / 'device_info'])
        metadata['device_info'] = run([WORK / 'device_info'])
        (RESULTS / 'device-info.txt').write_text(metadata['device_info'])
        nvprof = shutil.which('nvprof') or str(Path(args.nvcc).with_name('nvprof'))
        for n in args.sizes:
            print(f'n={n}: 5 конфигураций, прогрев {args.warmup} на min(n, {args.warmup_size}) элементах', flush=True)
            text = run([WORK / 'benchmark', n, args.repeats, args.warmup, args.warmup_size])
            rows = list(csv.DictReader(io.StringIO(text)))
            assert [(int(r['grid']), int(r['block'])) for r in rows] == CONFIGS
            for row in rows:
                assert int(row['n']) == n and int(row['errors']) == 0
                assert int(row['gpu_checks']) == args.warmup + args.repeats
                assert int(row['warmup']) == args.warmup and int(row['repeats']) == args.repeats
                assert all(math.isfinite(float(row[k])) and float(row[k]) > 0 for k in ['cpu_ms', 'gpu_ms', 'speedup'])
                assert math.isclose(float(row['speedup']), float(row['cpu_ms']) / float(row['gpu_ms']), rel_tol=1e-9)
            filename = f'benchmark-{n}.csv'
            (RESULTS / filename).write_text(text)
            metadata['result_files'].append(filename)
            if args.profile:
                for grid, block in CONFIGS:
                    if n == 128000000 and grid == 1 and not args.profile_slow:
                        continue
                    filename = f'nvprof-{n}-{grid}-{block}.log'
                    print(f'  nvprof: <<<{grid}, {block}>>>', flush=True)
                    output = run([nvprof, '--profile-from-start', 'off', '--log-file', RESULTS / filename,
                                  WORK / 'benchmark', n, 1, args.warmup, args.warmup_size, '--profile', grid, block])
                    assert 'Profile passed' in output
                    text = (RESULTS / filename).read_text().replace(str(LAB), '.')
                    (RESULTS / filename).write_text(text)
                    profile = parse_profile(text)
                    kernels = [r for r in profile if r['section'] == 'GPU activities']
                    assert {r['name'] for r in kernels} >= {'scan_blocks', 'scatter'}
                    assert all(r['ms'] > 0 for r in kernels)
                    metadata['profile_files'].append(filename)
        metadata['tests'] = {mode: run(['python3', '-B', LAB / 'tests/check.py', WORK / binary]).strip()
                             for mode, binary in [('cpu', 'cpu'), ('gpu', 'lab5')]}
        metadata.update(status='complete', profile_status=(
            'complete' if len(metadata['profile_files']) == len(args.sizes) * len(CONFIGS) else 'partial'
        ) if args.profile else 'not_requested')
        return 0
    except Exception as error:
        metadata.update(status='failed', error=str(error).replace(str(LAB), '.'))
        print(metadata['error'], flush=True)
        return 1
    finally:
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (RESULTS / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
