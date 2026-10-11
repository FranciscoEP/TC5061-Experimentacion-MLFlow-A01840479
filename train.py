"""Entrenamiento reproducible del baseline de cancelaciones de hoteles."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tc5061-matplotlib')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
import hashlib
import json
import random
import tempfile
from urllib.parse import unquote
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'TC5061_Hoteles_Logistica_Experimentacion'
NUMERIC = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
           'adults', 'children', 'babies', 'is_repeated_guest',
           'previous_cancellations', 'previous_bookings_not_canceled',
           'booking_changes', 'days_in_waiting_list', 'adr',
           'total_of_special_requests', 'has_agent', 'has_company']
CATEGORICAL = ['hotel', 'arrival_date_month', 'meal', 'market_segment',
               'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/hotel_bookings.csv', help='CSV original de Hotel booking demand.')
    parser.add_argument('--C', type=float, default=1.0, help='Inversa de regularización L2; debe ser positiva.')
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none', help='Ponderación de clases durante el ajuste.')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerancia positiva de convergencia del optimizador.')
    parser.add_argument('--max-iter', type=int, default=1000, help='Límite positivo de iteraciones.')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para particiones y modelo.')
    parser.add_argument('--experiment', default=EXPERIMENT, help='Nombre del experimento MLflow.')
    parser.add_argument('--artifact-location', default=None, help='URI de artefactos para experimentos nuevos; SQLite usa mlartifacts junto a su base de datos.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results', help='Carpeta para resúmenes JSON de las corridas.')
    parser.add_argument('--run-name', default='logistica_default', help='Nombre legible de la corrida.')
    parser.add_argument('--tracking-uri', default=f'sqlite:///{ROOT / "mlflow.db"}', help='URI de MLflow Tracking.')
    args = parser.parse_args()
    if not np.isfinite(args.C) or not np.isfinite(args.tol) or args.C <= 0 or args.tol <= 0 or args.max_iter <= 0:
        parser.error('C, tol y max-iter deben ser positivos y finitos.')
    if not 0 <= args.seed < 2**32:
        parser.error('seed debe estar entre 0 y 2**32 - 1.')
    if not args.data.is_file():
        parser.error(f'No existe {args.data}. Consulta README.md para obtener el CSV.')
    return args


def load_and_clean(path):
    """Replica las reglas del notebook antes de dividir los datos."""
    raw = pd.read_csv(path)
    required = set(NUMERIC + CATEGORICAL + ['is_canceled', 'agent', 'company']) - {'has_agent', 'has_company'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Columnas faltantes: {sorted(missing)}')
    if raw.is_canceled.isna().any() or set(raw.is_canceled.unique()) != {0, 1}:
        raise ValueError('is_canceled debe contener las dos clases 0 y 1 sin nulos.')
    df = raw.drop_duplicates().copy()
    duplicates = len(raw) - len(df)
    df['children'] = df['children'].fillna(0).astype(int)
    df['has_agent'] = df['agent'].notna().astype(int)
    df['has_company'] = df['company'].notna().astype(int)
    df['meal'] = df['meal'].replace('Undefined', 'SC')
    keep = ((df.market_segment != 'Undefined') & (df.distribution_channel != 'Undefined')
            & ((df.adults + df.children + df.babies) > 0) & df.adr.between(0, 1000)
            & (df.adults <= 10) & (df.children < 10) & (df.babies < 9))
    df = df.loc[keep].reset_index(drop=True)
    return df, {'raw_rows': len(raw), 'duplicates_removed': duplicates, 'clean_rows': len(df),
                'other_rows_removed': len(raw) - duplicates - len(df)}


def split_data(df, seed):
    """60/20/20 estratificado; prueba conserva la partición original de 20%."""
    X, y = df[NUMERIC + CATEGORICAL], df.is_canceled
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(args):
    preprocessing = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])
    return Pipeline([('prepare', preprocessing), ('classify', LogisticRegression(
        C=args.C, class_weight=None if args.class_weight == 'none' else args.class_weight,
        tol=args.tol, max_iter=args.max_iter, random_state=args.seed, solver='lbfgs'))])


def evaluate(model, X, y):
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return {'accuracy': float(accuracy_score(y, predicted)),
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probability))}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    df, counts = load_and_clean(args.data)
    X_train, X_val, _, y_train, y_val, _ = split_data(df, args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    # SQLite almacena artefactos junto a su base; servidores remotos deciden su ubicación.
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        location = args.artifact_location
        if location is None and args.tracking_uri.startswith('sqlite:///'):
            database = unquote(args.tracking_uri[len('sqlite:///'):].split('?', 1)[0])
            if database != ':memory:':
                location = (Path(database).expanduser().resolve().parent / 'mlartifacts').as_uri()
        mlflow.create_experiment(args.experiment, artifact_location=location)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        params = {'C': args.C, 'class_weight': args.class_weight, 'tol': args.tol,
                  'max_iter': args.max_iter, 'seed': args.seed, 'solver': 'lbfgs',
                  'train_fraction': .6, 'validation_fraction': .2, 'test_fraction': .2,
                  'threshold': .5, 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  'train_rows': len(X_train), 'validation_rows': len(X_val), **counts}
        for name, value in params.items():
            mlflow.log_param(name, value)
        mlflow.set_tags({'stage': 'validation', 'selection_metric': 'val_f1', 'data_source': 'Kaggle/jessemostipak/hotel-booking-demand'})
        with threadpool_limits(limits=1):
            model = build_model(args)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val)
        for name, value in metrics.items():
            mlflow.log_metric(f'val_{name}', value)
        iterations = int(model.named_steps['classify'].n_iter_[0])
        mlflow.log_metric('n_iter', iterations)
        mlflow.set_tag('converged', str(iterations < args.max_iter).lower())
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            (temp / 'split_indices.json').write_text(json.dumps({'train': X_train.index.tolist(), 'validation': X_val.index.tolist()}))
            ConfusionMatrixDisplay.from_predictions(y_val, model.predict(X_val), display_labels=['No cancelada', 'Cancelada'], cmap='Blues', colorbar=False)
            plt.title(f'Validación: {args.run_name}')
            plt.tight_layout()
            plt.savefig(temp / 'confusion_matrix.png', dpi=160)
            plt.close()
            for artifact in temp.iterdir():
                mlflow.log_artifact(str(artifact), artifact_path='evaluation')
        mlflow.log_artifact(str(Path(__file__)), artifact_path='source')
        mlflow.log_artifact(str(ROOT / 'requirements.txt'), artifact_path='source')
        info = mlflow.sklearn.log_model(model, name='pipeline', serialization_format='cloudpickle', input_example=X_train.head(3),
                                       pip_requirements=str(ROOT / 'requirements.txt'))
        result = {'run_id': run.info.run_id, 'run_name': args.run_name, 'model_uri': info.model_uri,
                  'params': params, 'metrics': metrics, 'n_iter': iterations}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f'{run.info.run_id}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
