# TC5061 - Experimentación con MLflow

Francisco Vicente Eugenio Ponce · A01840479

Repositorio individual: [TC5061-Experimentacion-MLFlow-A01840479](https://github.com/FranciscoEP/TC5061-Experimentacion-MLFlow-A01840479).

Predicción de cancelaciones de reservas. Migración del baseline de regresión logística de la semana 03. El notebook se conserva únicamente para exploración y revisión de limpieza.

## Requisitos e instalación

El proyecto usa **Python 3.13 + `venv` + `pip`**. `venv` crea el entorno virtual `.venv` para aislar las dependencias; `pip` instala las versiones indicadas en `requirements.txt`. Ambos vienen con una instalación estándar de Python. **No se requiere `uv`.**

Antes de empezar, instala Git y Python 3.13. Necesitas Internet para instalar las dependencias y descargar el CSV. El entorno del informe usó Python 3.13.5. El entrenamiento se ejecuta desde la terminal; no requiere Jupyter, un servidor externo de MLflow ni una GPU.

### 1. Obtener el proyecto

```bash
git clone https://github.com/FranciscoEP/TC5061-Experimentacion-MLFlow-A01840479.git
cd TC5061-Experimentacion-MLFlow-A01840479
```

Si ya tienes una copia local, omite la clonación y entra a la carpeta donde están `train.py` y `requirements.txt`.

### 2. Crear el entorno e instalar las dependencias

Elige los comandos de tu sistema operativo. Se ejecutan desde la raíz del proyecto.

**macOS o Linux:**

```bash
python3.13 --version
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

El primer comando debe mostrar `Python 3.13.x`. Si no existe `python3.13` o `py -3.13`, instala Python 3.13 antes de continuar. Al activar el entorno suele aparecer `(.venv)` al inicio de la terminal. Los comandos `python` y `python -m pip` siguientes usarán ese entorno.

Si PowerShell no permite activar el entorno, puedes ejecutar los comandos directamente con su Python, sin cambiar la configuración del sistema. Por ejemplo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

En ese caso sustituye `python` por `.\.venv\Scripts\python.exe` en los demás comandos de esta guía.

La creación del entorno y la instalación se hacen una vez por copia del proyecto. **Cada vez que abras otra terminal**, entra a la raíz del proyecto y vuelve a activar `.venv` con el comando de tu sistema operativo. No necesitas crear el entorno otra vez.

Todos los pasos siguientes se ejecutan desde esa misma carpeta, con `.venv` activo. `requirements.txt` contiene las dependencias de entrenamiento; `requirements-report.txt` solo se utiliza para regenerar el PDF.

## Datos

Descarga y descomprime `hotel_bookings.csv` del [dataset original en Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Kaggle puede pedir iniciar sesión. Crea la carpeta y coloca ahí el CSV:

```bash
python -c "from pathlib import Path; Path('data').mkdir(exist_ok=True)"
```

Ubicación final:

```text
data/hotel_bookings.csv
```

El CSV se excluye de Git. Verifica que no tenga un nombre como `hotel_bookings (1).csv`. Para reproducir el informe, el SHA-256 debe ser `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`:

```bash
python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('data/hotel_bookings.csv').read_bytes()).hexdigest())"
```

## Ejecutar una corrida

Para comprobar la ejecución con valores por defecto:

```bash
python train.py
```

Al finalizar se imprime un JSON con el ID, los parámetros y las métricas de la corrida. Se crean `mlflow.db`, los artefactos/modelo en `mlartifacts/` y un JSON en `results/`. La corrida se llama `logistica_default`.

Para consultar la ayuda y cambiar hiperparámetros sin editar el código:

```bash
python train.py --help
python train.py --C 1 --class-weight balanced --tol 0.0001 --max-iter 1000 --seed 42 --run-name balanced_manual
```

## Reproducir el trabajo completo

Ejecuta estos comandos **en este orden**:

```bash
python run_experiments.py
python finalize_results.py
```

`run_experiments.py` entrena siete configuraciones distintas y una repetición del baseline, bajo el experimento `TC5061_Hoteles_Logistica_Experimentacion`. `finalize_results.py` selecciona el máximo F1 de validación (desempate por orden del lote), comprueba la repetición y evalúa el modelo elegido en prueba sin reentrenarlo. Genera `results/comparison.csv`, `results/summary.json` y `results/test_metrics.json`.

Resultados de referencia del informe: `04_balanced`, F1 de validación **0.565116**, F1 de prueba **0.566508**, semilla **42** y 87,208 filas limpias (52,324 / 17,442 / 17,442). En `results/summary.json` puedes consultar la comprobación de repetibilidad; los parámetros, las métricas, los artefactos y los modelos se inspeccionan en MLflow como se indica abajo.

Cada ejecución del lote añade ocho corridas y actualiza `results/batch.json`; ejecuta la finalización después de cada lote. Los IDs nuevos no serán los del informe. Los JSON/CSV versionados son evidencia de la entrega, pero la base y los modelos no se suben a Git: una copia nueva debe ejecutar ambos scripts para regenerar el seguimiento local.

## Inspeccionar los resultados en MLflow

Después de ejecutar el lote, inicia la interfaz:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Conserva esa terminal abierta y visita <http://127.0.0.1:5001>. La interfaz usa la base local creada por el entrenamiento. Detén el servidor con `Ctrl+C` cuando termines. Para salir del entorno virtual, ejecuta `deactivate`; los resultados y modelos se conservan en la carpeta del proyecto.

- Abre `TC5061_Hoteles_Logistica_Experimentacion` y revisa **Runs**. El filtro `attributes.status = 'FINISHED' AND tags.mlflow.runName LIKE '0%'` oculta las corridas manuales; si ejecutaste varios lotes, muestra también los anteriores. Los ocho IDs del lote actual están en `results/batch.json`.
- Selecciona esas ocho corridas y abre **Compare**. En **Parameters**, activa **Show diff only**; contrasta `C`, `class_weight` y `tol` con las métricas `val_f1`, `val_precision`, `val_recall` y `val_roc_auc`.
- Abre la corrida elegida `04_balanced`. En **Artifacts** revisa `evaluation/confusion_matrix.png`, `metrics.json`, `split_indices.json` y `test_metrics.json`, y la carpeta `source/` con código y dependencias.
- En **Models** inspecciona el pipeline registrado. Las métricas `test_*` se registran solo en la corrida elegida al finalizar.

Las capturas entregadas están en `evidence/`; las corridas que reproduzcas tienen IDs y fechas nuevos.

## Problemas frecuentes

| Mensaje o situación | Qué revisar |
|---|---|
| No existe `data/hotel_bookings.csv` | Descarga el CSV y colócalo con ese nombre en `data/`. |
| `ModuleNotFoundError` o versiones distintas | Activa `.venv` e instala `requirements.txt` con ese mismo `python`. |
| Falló una corrida del lote | Corrige la causa que muestra la terminal y repite el lote completo antes de finalizar. |
| Puerto 5001 ocupado | Usa `--port 5002` al abrir MLflow y visita `http://127.0.0.1:5002`. |
| Interfaz vacía | Abre MLflow desde la raíz del repositorio, con `sqlite:///mlflow.db`. |

## Diseño y parámetros

- `--C`: inversa de regularización L2, por defecto 1. Menor C implica mayor regularización.
- `--class-weight`: `none` (defecto) o `balanced`.
- `--tol`: tolerancia del optimizador, por defecto 0.0001.
- `--max-iter`: máximo de iteraciones, por defecto 1000.
- `--seed`: semilla, por defecto 42.
- `--data`, `--tracking-uri`, `--experiment`, `--run-name`: rutas y organización del seguimiento.
- `--artifact-location`: URI opcional para los artefactos de un experimento nuevo. Con SQLite, por defecto se guardan junto a la base de datos; con un servidor remoto, este decide la ubicación. Un experimento existente conserva su ubicación original.
- `--output-dir`: carpeta para resúmenes JSON; por defecto `results/`. Permite aislar las ejecuciones de prueba.

Se eliminan duplicados exactos, categorías indefinidas y casos fuera de los filtros del notebook: 119,390 → 87,208 filas. Se conservan las mismas 23 variables. País, estado final y fecha de estado, habitación asignada y estacionamiento se excluyen. No se transforman país, identificadores o fecha de estado innecesariamente porque no entran al pipeline; las banderas `has_agent` y `has_company` sí se derivan de los identificadores.

Se separa primero 20% para prueba y luego 25% del 80% restante para validación: 52,324 / 17,442 / 17,442. Ambas divisiones son estratificadas con semilla 42. El preprocesamiento se ajusta exclusivamente en entrenamiento. Se elige por F1 de validación; las métricas `test_*` aparecen solo en la corrida elegida. El 20% de prueba coincide con la partición del notebook anterior, por lo que no es un conjunto nunca observado históricamente; no se usa para seleccionar en esta actividad.

`run_experiments.py` compara seis combinaciones de C={0.01,1,100} y class_weight={none,balanced}, una tolerancia alternativa y la repetición del baseline. `max_iter`, solver y umbral permanecen constantes. Se fija la semilla de Python, NumPy, particiones y modelo, y se limita BLAS a un hilo. Se registra el hash del CSV y los índices de entrenamiento/validación. La igualdad exacta se verificó en el mismo entorno; otras plataformas o versiones pueden introducir diferencias numéricas.

## Evidencias y resultados

- `results/comparison.csv`: ocho corridas exitosas.
- `results/summary.json`: selección, prueba final y comprobación exacta de probabilidades.
- `results/batch.json`: IDs y configuraciones del lote.
- `evidence/`: capturas reales de la interfaz.
- `output/pdf/`: informe de entrega.
- `mlflow.db` y `mlartifacts/`: seguimiento local y modelos (excluidos de Git, se regeneran ejecutando los scripts).

En MLflow: Model training → experimento → Runs → filtrar `attributes.status = 'FINISHED'` → seleccionar corridas → Compare. En Parameters activa “Show diff only”. Revisa métricas `val_*`. Abre `04_balanced` → Artifacts para ver `evaluation/` y `source/`; en Models encontrarás el pipeline registrado. El seguimiento local original conserva una corrida fallida inicial por incompatibilidad del formato de serialización predeterminado; no forma parte del lote analizado ni aparece en una copia nueva. El código final guarda con `cloudpickle`; carga únicamente modelos propios o de origen confiable.

Resultado: `04_balanced` (C=1, balanced, tol=0.0001) obtuvo F1 de validación 0.565116 y F1 de prueba 0.566508. El ROC-AUC de validación cambia apenas de 0.769115 a 0.769633 respecto al baseline, sin mejora apreciable de discriminación. El balanceo cambia el entrenamiento y el equilibrio precisión-recall a umbral fijo de 0.5; no se asume que sea equivalente a cambiar únicamente el umbral. La precisión disminuye al mejorar recall: la elección depende del costo de falsas alarmas. No se valida aún con división temporal ni nuevas semillas. Las diferencias pequeñas entre valores de C no demuestran superioridad estadística.

## Documentación

- [MLflow sklearn: registro de modelo y firma](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
- [Regresión logística: parámetros](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## Regenerar el PDF

`build_report.py` utiliza ReportLab y las fuentes Arial de macOS (rutas declaradas en el script). Instala primero `python -m pip install -r requirements-report.txt` y ejecuta `python build_report.py`. El PDF incluye por defecto el enlace al repositorio individual indicado arriba; puedes cambiarlo con `--repo-url URL_DEL_REPOSITORIO`. La generación del PDF no forma parte del entorno mínimo de entrenamiento.

La corrida adicional `logistica_default` verifica la ejecución sin argumentos; no cambia el lote de ocho corridas reportado. Para mostrar únicamente ese lote, usa el filtro `attributes.status = 'FINISHED' AND tags.mlflow.runName LIKE '0%'`.
