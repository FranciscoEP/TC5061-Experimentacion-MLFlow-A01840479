"""Genera el informe con el repositorio individual; --repo-url permite cambiar el enlace."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, NextPageTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-url', default='https://github.com/FranciscoEP/TC5061-Experimentacion-MLFlow-A01840479', help='URL del repositorio individual de GitHub.')
args = parser.parse_args()
pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold')
body = ParagraphStyle('body',fontName='Arial',fontSize=11,leading=16.5,spaceAfter=9)
code = ParagraphStyle('code',parent=body,fontName='Arial',spaceAfter=3)
heading = ParagraphStyle('heading',parent=body,fontName='Arial-Bold',textColor=colors.HexColor('#193854'),spaceBefore=5,spaceAfter=10)
summary = json.loads((ROOT/'results/summary.json').read_text())
rows = list(csv.DictReader((ROOT/'results/comparison.csv').open()))
story=[]
def p(text, style=body): story.append(Paragraph(text,style))
def page(): story.append(PageBreak())
def shot(name,caption,width=500):
    p(caption,heading)
    im=Image(str(ROOT/'evidence'/name))
    im.drawWidth=width
    im.drawHeight=width*im.imageHeight/im.imageWidth
    story.append(im)

def footer(canvas,doc):
    canvas.setFont('Arial',11)
    canvas.setFillColor(colors.HexColor('#586577'))
    canvas.drawString(50,28,'TC5061 | Experimentación con MLflow | 4 de octubre de 2026')
    canvas.drawRightString(canvas._pagesize[0]-50,28,str(doc.page))

p('Código de experimentación base con MLflow',heading)
p('Francisco Vicente Eugenio Ponce · A01840479<br/>TC5061 · Actividad individual')
p('Análisis de resultados',heading)
p('Se migró la regresión logística del notebook a un script con funciones para carga, limpieza, partición, preprocesamiento, entrenamiento y evaluación. Se conservaron las 23 variables del baseline y sus filtros: 119,390 reservas originales se redujeron a 87,208. La imputación, el escalado y la codificación se ajustaron solo con entrenamiento.')
p('Se usaron 52,324 reservas para entrenamiento, 17,442 para validación y 17,442 para prueba, con estratificación y semilla 42. Las siete configuraciones distintas y una repetición se registraron bajo TC5061_Hoteles_Logistica_Experimentacion. El criterio de selección fue F1 de validación, con umbral fijo de 0.5; la prueba se consultó después de elegir el modelo.')
p('La mejor configuración por ese criterio fue <b>04_balanced: C=1, class_weight=balanced, tol=0.0001 y max_iter=1000</b>. Logró F1=0.5651, precisión=0.4698, recall=0.7090 y ROC-AUC=0.7696 en validación. El mismo modelo guardado, sin reentrenamiento, obtuvo F1=0.5665 y ROC-AUC=0.7770 en prueba.')
p('El hiperparámetro más influyente sobre F1 fue <b>class_weight</b>: al cambiar únicamente de none a balanced, F1 subió de 0.4689 a 0.5651 (+0.0962) y recall de 0.3499 a 0.7090, pero la precisión bajó de 0.7105 a 0.4698. El ROC-AUC apenas cambió (0.7691 a 0.7696): no se observó una mejora apreciable del ordenamiento de las predicciones. El beneficio aparece en el equilibrio precisión-recall a umbral 0.5. El balanceo modifica la función de pérdida y puede cambiar los coeficientes; no equivale necesariamente a mover solo el umbral.')
p('C tuvo un efecto pequeño en todo el rango 0.01-100: el rango de F1 fue 0.0107 sin balanceo y 0.0028 con balanceo. Aumentar tol a 0.01 redujo las iteraciones de 60 a 8; F1 subió 0.0071, pero ROC-AUC cayó 0.0064. No hubo una mejora general de las métricas.')
p('Las limitaciones son una sola partición y semilla, ausencia de validación temporal y posible información tardía en booking_changes o peticiones especiales. Eliminar duplicados puede excluir reservas grupales legítimas. La prueba coincide con la usada en la semana anterior: no es evidencia completamente nueva. Las diferencias pequeñas de C no demuestran superioridad estadística. El balanceo detecta más cancelaciones a costa de falsas alarmas; antes de usarlo se requieren costos de negocio y datos posteriores.')
page()
p('Reproducibilidad, ejecución y repositorio',heading)
if args.repo_url:
    url=escape(args.repo_url,{'"':'&quot;'})
    p(f'Repositorio individual: <link href="{url}" color="#175b91">{url}</link>')
else:
    p('<b>Pendiente para entrega:</b> incorporar la URL del repositorio individual de GitHub y verificar el acceso del profesor. Esta versión es un borrador completo de resultados y evidencias.')
p('Python 3.13.5; versiones fijadas en requirements.txt. Semilla 42 en Python, NumPy, las dos particiones y el modelo; cálculo numérico limitado a un hilo. La repetición obtuvo métricas y probabilidades exactamente iguales (diferencia máxima=0). Se verificaron particiones disjuntas y la carga del pipeline registrado. El hash SHA-256 del CSV y los índices de partición quedan registrados en MLflow.')
p('<b>Comandos desde la carpeta del proyecto</b> (tras instalar requirements.txt y colocar el CSV según README):')
for command in ['python train.py', 'python train.py --class-weight balanced', 'python run_experiments.py', 'python finalize_results.py', 'mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001']:
    p(command,code)
p('El segundo comando reproduce los hiperparámetros elegidos: los valores por defecto son C=1, tol=0.0001, max_iter=1000 y seed=42.')
p('Tabla de validación (métricas redondeadas)',heading)
data=[['Corrida','C','Peso','tol','F1','Recall','AUC']]
for r in rows:
    data.append([r['run'].split('_')[0],f"{float(r['C']):g}",r['class_weight'],f"{float(r['tol']):g}",*[f"{float(r[m]):.4f}" for m in ['f1','recall','roc_auc']]])
t=Table(data,colWidths=[53,48,74,68,70,70,70],hAlign='LEFT')
t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Arial'),('FONTSIZE',(0,0),(-1,-1),11),('LEADING',(0,0),(-1,-1),16.5),('FONTNAME',(0,0),(-1,0),'Arial-Bold'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf3')),('BACKGROUND',(0,4),(-1,4),colors.HexColor('#e7f2ee')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#93a8b8')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]))
story.append(t)
story.append(Spacer(1,9))
p('01: baseline; 02-03: C; 04-06: balanceo; 07: tolerancia; 08: repetición de 01. max_iter=1000 en todas. Las capturas de MLflow aparecen en los anexos.')
p('Fuentes: <link href="https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand" color="#175b91">Hotel booking demand (Kaggle)</link>; <link href="https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html" color="#175b91">MLflow sklearn</link>; <link href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" color="#175b91">LogisticRegression (scikit-learn)</link>.')
story.append(NextPageTemplate('landscape'))
page()
shot('runs_comparison_wide.jpg','Anexo A. Ocho corridas: hiperparámetros y métricas de validación',700)
p('Siete configuraciones diferentes y una repetición, agrupadas en el mismo experimento. La mayor F1 corresponde a 04_balanced; 01 y 08 tienen el mismo resultado.')
page()
shot('compare_parameters_wide.jpg','Anexo B. Vista Compare: detalles y parámetros de las ocho corridas',700)
p('Vista Compare con Run details y Parameters abiertos. De izquierda a derecha: 08, 07, 06, 05, 04, 03, 02 y 01. Show diff only muestra C, class_weight y tol; las métricas de esas mismas corridas se muestran junto a los parámetros en el anexo A.')
story.append(NextPageTemplate('portrait'))
page()
shot('run_artifacts.jpg','Anexo C. Detalle de 04_balanced y sus artefactos')
p('Matriz de confusión de validación: TN=8,804; FP=3,840; FN=1,396; TP=3,402. También se registraron métricas JSON, índices de partición, código, dependencias y el pipeline completo mediante mlflow.sklearn.log_model.')
output=ROOT/'output/pdf/TC5061_Experimentacion_MLFlow_EugenioFrancisco.pdf'
doc = BaseDocTemplate(str(output), pagesize=letter, title='TC5061 - Experimentación con MLflow', author='Francisco Vicente Eugenio Ponce')
doc.addPageTemplates([
    PageTemplate(id='portrait', frames=[Frame(50,45,512,705,id='portrait-frame')], pagesize=letter, onPage=footer),
    PageTemplate(id='landscape', frames=[Frame(36,42,720,528,id='landscape-frame')], pagesize=landscape(letter), onPage=footer),
])
doc.build(story)
print(output)
