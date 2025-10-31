## Melanoma Classification with XAI

Proyecto de **clasificación de imágenes dermatológicas** para detectar *melanoma vs no melanoma* usando el dataset **HAM10000** y el modelo **EfficientNetV2-S** preentrenado en *ImageNet*.  
El enfoque combina *fine-tuning* con estrategias para mitigar el desbalance (data augmentation, class weights y ajuste de umbral orientado a recall) e incluye técnicas **XAI (Explainable AI)** para interpretar visualmente las predicciones.

## Plan de fine-tuning con EfficientNetV2-S

**Objetivo:** adaptar un modelo preentrenado en ImageNet (EfficientNetV2-S) para clasificar *melanoma vs no melanoma* en HAM10000.

**Pasos principales:**
1. Cargar EfficientNetV2-S sin la “top” (`include_top=False`) y con pesos de ImageNet.  
2. Congelar el backbone y entrenar solo la cabeza nueva  
   (GAP → Dense(128, ReLU, Dropout) → Dense(1, Sigmoid)) con `class_weight`.  
3. Descongelar parcialmente las últimas capas del backbone y hacer *fine-tuning* con LR bajo.  
4. Evaluar con **ROC-AUC**, **PR-AUC** y ajustar **umbral** para priorizar *recall*.  
5. Guardar pesos y registrar seeds/versiones.

## Por qué usamos **HAM10000** (y no un dataset más balanceado)

Aunque HAM10000 presenta un fuerte desbalance (≈11% melanomas vs 89% no melanomas), es el **dataset PÚBLICO más completo, limpio y validado** para el diagnóstico automatizado de lesiones cutáneas.  
Fue curado por especialistas dermatólogos, se utiliza en benchmarks internacionales (ISIC Challenge) y refleja **la prevalencia real** de melanoma en la práctica clínica.

Cambiarlo por un dataset artificialmente balanceado (como PH2 o subconjuntos de ISIC) reduciría la **diversidad visual** y haría que el modelo aprenda un escenario poco realista.  
En cambio, mantener HAM10000 y aplicar **estrategias de mitigación de desbalance** —*data augmentation*, *class weights* y *ajuste de umbral enfocado en recall*— permite:

- **Preservar el realismo clínico**, entrenando sobre una distribución que imita la ocurrencia real del melanoma.  
- **Corregir el sesgo estadístico** sin distorsionar los datos originales.  
- **Mejorar la generalización**: el modelo aprende patrones visuales robustos en lugar de memorizar ejemplos repetidos.  
- **Alinear los objetivos clínicos**: maximizar *sensibilidad (recall)* para reducir falsos negativos.

> En resumen: HAM10000 no solo es un dataset accesible y reproducible,  
> sino el que garantiza que nuestro modelo aprenda bajo condiciones **realistas, éticas y clínicamente relevantes**.


## Estructura y progreso del repositorio

| Etapa | Descripción | Estado |
|-------|--------------|--------|
| Selección del modelo | EfficientNetV2-S (preentrenado en ImageNet) | ok |
| Descarga del dataset | HAM10000 desde Kaggle | ok |
| Exploración y limpieza | EDA, balance, análisis demográfico | ok |
| Preparación de imágenes | Resize 224×224, normalización, división train/val/test | ok |
| Data augmentation | Rotación ±15°, zoom 0.9-1.1, flip horizontal | ok |
| Class weights | Cálculo automático con `sklearn.utils.class_weight` | soon |
| Fine-tuning | Entrenamiento con LR bajo, descongelar capas superiores | soon |
| Evaluación | ROC-AUC, PR-AUC, matriz de confusión, ajuste de umbral | soon |
| XAI | Grad-CAM e interpretación de activaciones | soon |
| Documentación | README, notebook limpio y visualizaciones | not yet |
