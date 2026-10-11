# TC5061 - Experimentación con MLflow

Francisco Vicente Eugenio Ponce · A01840479

[Repositorio individual](https://github.com/FranciscoEP/TC5061-Experimentacion-MLFlow-A01840479)

El proyecto predice cancelaciones de reservas con una regresión logística. El modelo de la semana 3 se trasladó a un script para cambiar hiperparámetros desde la terminal y comparar las corridas en MLflow. El notebook de esta entrega conserva únicamente la exploración y la revisión de limpieza; el original completo se conserva en la actividad de semana 3.

## 1. Instalación

Necesitas Git y Python **3.13**. Se usa `venv` para crear el entorno y `pip` para instalar las dependencias; **no se requiere `uv`**. El informe se generó con Python 3.13.5.

```bash
git clone https://github.com/FranciscoEP/TC5061-Experimentacion-MLFlow-A01840479.git
cd TC5061-Experimentacion-MLFlow-A01840479
```

Si ya tienes el proyecto, entra a la carpeta que contiene `train.py` y `requirements.txt`.

**macOS o Linux:**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows, en PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Todos los comandos siguientes se ejecutan desde la raíz del proyecto, con `.venv` activo. Al abrir otra terminal, entra a esa carpeta y vuelve a activar el entorno; no necesitas crearlo de nuevo. Si PowerShell bloquea la activación, sustituye `python` por `.\.venv\Scripts\python.exe` en los comandos.

## 2. Datos

Descarga y descomprime `hotel_bookings.csv` de [Hotel booking demand en Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). Kaggle puede pedir iniciar sesión. Crea la carpeta `data` y coloca ahí el archivo:

```bash
python -c "from pathlib import Path; Path('data').mkdir(exist_ok=True)"
```

La ubicación debe ser **`data/hotel_bookings.csv`**. El CSV no se incluye en Git. El archivo utilizado tiene SHA-256 `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`; el script registra el hash en cada corrida.

## 3. Ejecutar el entrenamiento

Una corrida con valores por defecto:

```bash
python train.py
```

Consultar la ayuda y ejecutar una configuración distinta:

```bash
python train.py --help
python train.py --C 1 --class-weight balanced --tol 0.0001 --max-iter 1000 --seed 42 --run-name balanced_manual
```

| Argumento | Qué cambia | Valor por defecto |
|---|---|---|
| `--C` | Inversa de la regularización: un valor menor regulariza más. | `1` |
| `--class-weight` | Ponderación de clases: `none` o `balanced`. | `none` |
| `--tol` | Tolerancia de convergencia del optimizador. | `0.0001` |
| `--max-iter` | Límite de iteraciones del entrenamiento. | `1000` |
| `--seed` | Semilla de las particiones y el modelo. | `42` |

Al terminar se imprime un JSON con los parámetros y métricas. MLflow guarda la corrida en `mlflow.db`, y el modelo y sus artefactos en `mlartifacts/`.

## 4. Reproducir las ocho corridas

Ejecuta en este orden:

```bash
python run_experiments.py
python finalize_results.py
```

El primer script prueba siete configuraciones y repite el baseline. El segundo compara F1 de validación, comprueba la repetición y evalúa el modelo elegido en prueba. Cada nuevo lote añade ocho corridas y actualiza los archivos de `results/`; ejecuta la finalización después de cada lote.

Se usa semilla 42 en Python, NumPy, las particiones y el modelo. Los datos se dividen en 60% entrenamiento, 20% validación y 20% prueba; el preprocesamiento se ajusta solo con entrenamiento. Las dependencias tienen versiones fijadas en `requirements.txt`. La repetición produjo métricas y probabilidades iguales en el mismo entorno; otras plataformas pueden introducir diferencias numéricas.

Resultado de referencia: **`04_balanced`**, con F1 de validación **0.565116** y F1 de prueba **0.566508**. El criterio de selección es F1; el aumento de recall viene acompañado de más falsas alarmas. Las limitaciones se explican en el PDF.

## 5. Revisar y comparar en MLflow

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Abre <http://127.0.0.1:5001> y conserva la terminal abierta.

1. Abre el experimento **`TC5061_Hoteles_Logistica_Experimentacion`**.
2. En **Runs**, selecciona las ocho corridas del lote y abre **Compare**. Sus IDs están en `results/batch.json`. El filtro `attributes.status = 'FINISHED' AND tags.mlflow.runName LIKE '0%'` oculta las corridas manuales, pero incluye lotes anteriores si los hay.
3. Compara `C`, `class_weight` y `tol` con las métricas `val_f1`, `val_precision`, `val_recall` y `val_roc_auc`. En **Parameters** puedes activar **Show diff only**.
4. Abre `04_balanced` y revisa **Artifacts**: matriz de confusión, métricas JSON, índices de partición y código. En **Models** está el pipeline registrado. Las métricas `test_*` se añaden al modelo elegido al finalizar el lote.

Detén MLflow con `Ctrl+C`. Para salir del entorno virtual, ejecuta `deactivate`.

## Archivos de la entrega

- `train.py`: carga, limpieza, preprocesamiento, entrenamiento, evaluación y registro.
- `run_experiments.py` y `finalize_results.py`: ejecución y comparación del lote.
- `TC5061_Notebook_Baseline_EugenioFrancisco.ipynb`: exploración y revisión de limpieza.
- `requirements.txt`: dependencias con versiones.
- `results/`: tabla comparativa, configuraciones y resumen de resultados.
- `evidence/`: capturas de MLflow.
- `output/pdf/`: informe para Canvas.

La base y los modelos locales no se incluyen en Git: una copia nueva debe ejecutar el lote y la finalización para regenerarlos. Los IDs y fechas serán distintos de los del informe.

Si falta el CSV, revisa su ubicación. Si falta una librería, activa `.venv` e instala `requirements.txt`. Si el puerto 5001 está ocupado, usa `--port 5002` y abre <http://127.0.0.1:5002>.

Para regenerar el PDF en macOS, instala `requirements-report.txt` y ejecuta `python build_report.py`; el generador utiliza las fuentes Arial de macOS y el enlace de GitHub indicado arriba. Este paso es opcional para ejecutar los experimentos.
