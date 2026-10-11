"""Ejecuta siete configuraciones y repite el baseline con la misma semilla."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# Cada entrada contiene el nombre de la corrida y sus cambios respecto al valor por defecto.
CONFIGS = [
    ('01_baseline', []),
    ('02_C_001', ['--C', '0.01']),
    ('03_C_100', ['--C', '100']),
    ('04_balanced', ['--class-weight', 'balanced']),
    ('05_balanced_C_001', ['--class-weight', 'balanced', '--C', '0.01']),
    ('06_balanced_C_100', ['--class-weight', 'balanced', '--C', '100']),
    ('07_tol_001', ['--tol', '0.01']),
    ('08_repeat_baseline', []),
]

if __name__ == '__main__':
    # Se reúnen únicamente los JSON creados por este lote.
    previous = set((ROOT / 'results').glob('*.json'))
    for name, args in CONFIGS:
        subprocess.run([sys.executable, str(ROOT / 'train.py'), '--run-name', name, *args], check=True)
    runs = [json.loads(p.read_text()) for p in set((ROOT / 'results').glob('*.json')) - previous]
    runs = [r for r in runs if 'run_name' in r]
    runs.sort(key=lambda r: r['run_name'])
    if len(runs) != len(CONFIGS):
        raise RuntimeError('No se encontraron todas las corridas de esta ejecución.')
    (ROOT / 'results' / 'batch.json').write_text(json.dumps(runs, indent=2))
