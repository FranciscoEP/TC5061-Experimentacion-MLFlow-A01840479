"""Compara las corridas, comprueba la repetición y evalúa el modelo elegido en prueba."""
import train  # Configura hilos antes de importar librerías numéricas.
import json
import hashlib
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = train.ROOT

def main():
    runs = json.loads((ROOT / 'results/batch.json').read_text())
    # La primera y la última corrida usan los mismos parámetros.
    base, repeat = runs[0], runs[-1]
    assert base['params'] == repeat['params']
    assert base['metrics'] == repeat['metrics'], 'Métricas distintas al repetir.'
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
    data_path = ROOT / 'data/hotel_bookings.csv'
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert all(r['params']['data_sha256'] == data_hash for r in runs), 'El CSV cambió desde el entrenamiento.'
    df, _ = train.load_and_clean(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = train.split_data(df, base['params']['seed'])
    assert not (set(X_train.index) & set(X_val.index) or set(X_train.index) & set(X_test.index) or set(X_val.index) & set(X_test.index))
    baseline_models = [mlflow.sklearn.load_model(r['model_uri']) for r in (base, repeat)]
    with threadpool_limits(limits=1):
        baseline_probabilities, repeated_probabilities = [m.predict_proba(X_val) for m in baseline_models]
    np.testing.assert_array_equal(baseline_probabilities, repeated_probabilities)
    # Se elige por F1 de validación; la repetición no participa en la selección.
    candidates = runs[:-1]
    best = max(candidates, key=lambda r: r['metrics']['f1'])
    model = mlflow.sklearn.load_model(best['model_uri'])
    with threadpool_limits(limits=1):
        # Verifica también las métricas del modelo deserializado.
        assert train.evaluate(model, X_val, y_val) == best['metrics']
        test_metrics = train.evaluate(model, X_test, y_test)
    with mlflow.start_run(run_id=best['run_id']):
        mlflow.set_tag('selected_by', 'maximum validation F1 among seven configurations')
        for name, value in test_metrics.items():
            mlflow.log_metric('test_' + name, value)
        (ROOT / 'results/test_metrics.json').write_text(json.dumps(test_metrics, indent=2))
        mlflow.log_artifact(str(ROOT / 'results/test_metrics.json'), artifact_path='evaluation')
    rows = [{'run': r['run_name'], 'run_id': r['run_id'], 'C': r['params']['C'],
             'class_weight': r['params']['class_weight'], 'tol': r['params']['tol'],
             'n_iter': r['n_iter'], **r['metrics']} for r in runs]
    pd.DataFrame(rows).to_csv(ROOT / 'results/comparison.csv', index=False)
    summary = {'best': best, 'test_metrics': test_metrics, 'test_rows': len(X_test),
               'reproducibility': {'metrics_equal': True, 'probabilities_exactly_equal': True,
                                   'max_absolute_difference': float(np.max(np.abs(baseline_probabilities - repeated_probabilities))),
                                   'validation_probabilities_sha256': hashlib.sha256(baseline_probabilities.tobytes()).hexdigest()},
               'split_disjoint': True}
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).drop(columns='run_id').to_string(index=False))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
