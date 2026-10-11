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
    canvas.drawString(50,28,'TC5061 | Experimentación con MLflow | 10 de octubre de 2026')
    canvas.drawRightString(canvas._pagesize[0]-50,28,str(doc.page))

p('Código de experimentación base con MLflow',heading)
p('Francisco Vicente Eugenio Ponce · A01840479<br/>TC5061 · Actividad individual')
p('Análisis de resultados',heading)
p('Para esta actividad se tomó la regresión logística del notebook de semana 3 y se pasó a train.py. Se conservaron las 23 variables y las reglas de limpieza: de 119,390 reservas quedaron 87,208. El preprocesamiento se ajustó solo con los datos de entrenamiento.')
p('Se usaron 52,324 reservas para entrenamiento, 17,442 para validación y 17,442 para prueba, con semilla 42. En el experimento TC5061_Hoteles_Logistica_Experimentacion de MLflow se registraron siete configuraciones distintas y una repetición del baseline. Se comparó el efecto de C, class_weight y tol, manteniendo max_iter=1000 y un umbral de 0.5. La selección se hizo con F1 de validación.')
p('La mejor configuración fue <b>04_balanced: C=1, class_weight=balanced, tol=0.0001 y max_iter=1000</b>. En validación obtuvo F1=0.5651, precisión=0.4698, recall=0.7090 y ROC-AUC=0.7696. Después de elegirla, el modelo guardado se evaluó en prueba sin volver a entrenarlo: F1=0.5665 y ROC-AUC=0.7770.')
p('El parámetro que más influyó fue <b>class_weight</b>. Al pasar de none a balanced con los demás valores iguales, F1 subió de 0.4689 a 0.5651 y recall de 0.3499 a 0.7090. Esto permitió detectar más cancelaciones, pero la precisión bajó de 0.7105 a 0.4698: también hubo más falsas alarmas. ROC-AUC cambió muy poco, de 0.7691 a 0.7696, por lo que la mejora de F1 no implica que todas las métricas hayan mejorado. El balanceo cambia el peso de las clases en la función de pérdida durante el entrenamiento; no equivale a mover únicamente el umbral de decisión.')
p('Cambiar C entre 0.01 y 100 tuvo un efecto menor: F1 estuvo entre 0.4587 y 0.4694 sin balanceo, y entre 0.5623 y 0.5651 con balanceo. Al usar tol=0.01, el entrenamiento pasó de 60 a 8 iteraciones; F1 subió a 0.4760, pero ROC-AUC bajó a 0.7627. Su F1 fue menor que el de la configuración elegida.')
p('Las pequeñas diferencias entre valores de C no demuestran superioridad estadística. Las principales limitaciones son usar una sola partición y semilla, y no probar con reservas de periodos posteriores. Algunas variables, como booking_changes, podrían contener información conocida después de la reserva. Además, quitar duplicados puede eliminar reservas grupales legítimas. La partición de prueba ya se había usado en semana 3, aunque aquí no se utilizó para elegir el modelo. Antes de usarlo con un hotel sería necesario revisar estas decisiones y el costo de las falsas alarmas.')
page()
p('Reproducibilidad, ejecución y repositorio',heading)
if args.repo_url:
    url=escape(args.repo_url,{'"':'&quot;'})
    p(f'Repositorio individual: <link href="{url}" color="#175b91">{url}</link>')
else:
    p('<b>Pendiente para entrega:</b> incorporar la URL del repositorio individual de GitHub y verificar el acceso del profesor. Esta versión es un borrador completo de resultados y evidencias.')
p('Se utilizó Python 3.13.5 y las versiones de requirements.txt. Se fijó semilla 42 en Python, NumPy, las particiones y el modelo, con un solo hilo de cálculo. La repetición produjo métricas y probabilidades idénticas (diferencia máxima=0), documentadas en results/summary.json. El SHA-256 del CSV se registra en MLflow para comprobar los datos utilizados.')
p('Preparación y datos: seguir los pasos del README desde la raíz del proyecto, con el entorno virtual activo.')
p('<b>Entrenamiento y comparación:</b>')
for command in ['python train.py', 'python train.py --C 1 --class-weight balanced --tol 0.0001 --max-iter 1000 --seed 42', 'python run_experiments.py', 'python finalize_results.py', 'python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001']:
    p(command,code)
p('El segundo comando usa la configuración elegida. Para reproducir el lote, ejecutar run_experiments.py y después finalize_results.py. Abrir MLflow en http://127.0.0.1:5001.')
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
p('Las ocho corridas pertenecen al mismo experimento. 04_balanced tiene el mayor F1; 01 y 08 repiten la misma configuración y resultado.')
page()
shot('compare_parameters_wide.jpg','Anexo B. Vista Compare: detalles y parámetros de las ocho corridas',700)
p('La vista Compare muestra los cambios de C, class_weight y tol. Las columnas corresponden a las corridas 08 a 01; sus métricas aparecen en el anexo A.')
story.append(NextPageTemplate('portrait'))
page()
shot('run_artifacts.jpg','Anexo C. Detalle de 04_balanced y sus artefactos')
p('La matriz de validación muestra 8,804 reservas no canceladas y 3,402 canceladas clasificadas correctamente, además de 3,840 falsas alarmas y 1,396 cancelaciones no detectadas. Se guardaron los artefactos y el pipeline completo con mlflow.sklearn.log_model.')
output=ROOT/'output/pdf/TC5061_Experimentacion_MLFlow_EugenioFrancisco.pdf'
doc = BaseDocTemplate(str(output), pagesize=letter, title='TC5061 - Experimentación con MLflow', author='Francisco Vicente Eugenio Ponce')
doc.addPageTemplates([
    PageTemplate(id='portrait', frames=[Frame(50,45,512,705,id='portrait-frame')], pagesize=letter, onPage=footer),
    PageTemplate(id='landscape', frames=[Frame(36,42,720,528,id='landscape-frame')], pagesize=landscape(letter), onPage=footer),
])
doc.build(story)
print(output)
