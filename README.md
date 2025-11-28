## Melanoma Classification with XAI

Proyecto de **clasificación de imágenes dermatológicas** para detectar *melanoma vs no melanoma* usando el dataset **DermMel** (balanceado) y el modelo **EfficientNetV2-S** preentrenado en *ImageNet*.  
El enfoque utiliza *fine-tuning* con data augmentation y técnicas **XAI (Explainable AI)** mediante Grad-CAM para interpretar visualmente las predicciones del modelo.

## Modelos Pre-entrenados

Los modelos generados durante el entrenamiento están disponibles en Google Drive:

**[Descargar Modelos](https://drive.google.com/drive/folders/1bRGqCDRvz_jmBiH_UfcSPDOXESPMtaPW?usp=sharing)**

Incluye:
- `best_model_stage1.keras` - Modelo después de la primera etapa de fine-tuning
- `best_model_stage2.keras` - Modelo después del fine-tuning completo
- `final_melanoma_model.keras` - Modelo final optimizado

## Plan de fine-tuning con EfficientNetV2-S

**Objetivo:** adaptar un modelo preentrenado en ImageNet (EfficientNetV2-S) para clasificar *melanoma vs no melanoma* en DermMel.

**Pasos principales:**
1. Cargar EfficientNetV2-S sin la "top" (`include_top=False`) y con pesos de ImageNet.  
2. Congelar el backbone y entrenar solo la cabeza nueva  
   (GAP → Dense(128, ReLU, Dropout) → Dense(1, Sigmoid)).  
3. Descongelar parcialmente las últimas capas del backbone y hacer *fine-tuning* con LR bajo.  
4. Evaluar con **ROC-AUC**, **PR-AUC**, matriz de confusión y métricas clínicas.  
5. Aplicar **Grad-CAM** para validar que el modelo aprende patrones clínicamente relevantes.  
6. Guardar pesos y registrar seeds/versiones.

## Por qué usamos **DermMel** (Dataset Balanceado)

**DermMel** es un dataset balanceado de imágenes dermatológicas disponible en Kaggle, diseñado específicamente para clasificación binaria de melanoma.

### Características del Dataset

- **Balanceado**: Igual cantidad de melanomas y no melanomas (~5,341 por clase en train)
- **Pre-dividido**: train_sep (10,682) / valid (3,562) / test (3,561)
- **Organizado**: Estructura de carpetas por clase (Melanoma / NotMelanoma)
- **Formato**: Imágenes JPEG de alta calidad
- **Fuente**: Kaggle - `drscarlat/melanoma`

### Ventajas del Dataset Balanceado

A diferencia de HAM10000 (11% melanoma, 89% no melanoma), DermMel ofrece:

- **Entrenamiento simplificado**: No requiere class weights ni ajuste complejo de umbrales
- **Evaluación directa**: Métricas como accuracy son más representativas
- **Convergencia rápida**: Modelo aprende ambas clases equitativamente
- **Baseline claro**: Permite enfocarse en arquitectura y XAI sin gestionar desbalance

### Trade-off: Realismo Clínico

**Limitación**: El balance artificial (50/50) NO refleja la prevalencia real de melanoma (~1-2% en screening).

**Implicación**: El modelo está optimizado para datasets balanceados. En producción clínica real, sería necesario:
- Re-calibrar probabilidades según prevalencia real
- Ajustar umbral de decisión priorizando recall
- Validar en datos con distribución clínica realista

> **Conclusión**: DermMel es ideal para **prototipos académicos** y **desarrollo de XAI**,  
> pero modelos para uso clínico deberían entrenarse/validarse con distribuciones realistas como HAM10000.


## Estructura y progreso del repositorio

| Etapa | Descripción | Estado |
|-------|--------------|--------|
| Selección del modelo | EfficientNetV2-S (preentrenado en ImageNet) | Completado |
| Descarga del dataset | DermMel (balanceado) desde Kaggle | Completado |
| Exploración y limpieza | EDA, visualización de ejemplos, verificación de balance | Completado |
| Preparación de imágenes | Resize 224×224, normalización ImageNet | Completado |
| Data augmentation | Rotación ±15°, zoom 0.9-1.1, flip horizontal, brightness | Completado |
| Fine-tuning | Stage 1: cabeza, Stage 2: descongelar parcial del backbone | Completado |
| Evaluación | ROC-AUC, PR-AUC, matriz de confusión, métricas clínicas | Completado |
| XAI | Grad-CAM para TP/FN/FP/TN con interpretación clínica | Completado |
| Documentación | README, notebook completo y visualizaciones | Completado |

---

<details>
<summary><b>Etapa de Evaluación (Click para expandir)</b></summary>

### Métricas Implementadas

#### Métricas Generales
- **ROC-AUC**: Capacidad de discriminación del modelo
- **PR-AUC**: Rendimiento en clase desbalanceada (melanoma)

#### Métricas Estándar
- **Accuracy**: Proporción de predicciones correctas (relevante en dataset balanceado)
- **Sensibilidad (Recall)**: Proporción de melanomas detectados
- **Especificidad**: Proporción de no melanomas correctamente clasificados
- **Precisión (Precision)**: Proporción de predicciones positivas correctas
- **F1-Score**: Media armónica de precisión y recall

#### Matriz de Confusión
```
┌─────────────────────────┐
│  TN        FP           │  TN: Verdaderos Negativos
│  FN        TP           │  FP: Falsos Positivos
└─────────────────────────┘  FN: Falsos Negativos (CRÍTICO)
                             TP: Verdaderos Positivos
```

### Visualizaciones Generadas

1. **Curva ROC** (`roc_curve.png`)
   - Muestra trade-off entre TPR y FPR
   - Comparación con clasificador aleatorio

2. **Curva Precision-Recall** (`precision_recall_curve.png`)
   - Evaluación de trade-off precision vs recall
   - Baseline = 0.5 (dataset balanceado)

3. **Matrices de Confusión Comparativas** (`confusion_matrices_comparison.png`)
   - Umbral 0.5 (default) vs Umbral óptimo
   - Visualización del impacto del ajuste de umbral

4. **Análisis de Trade-off** (`threshold_tradeoff.png`)
   - Sensibilidad vs Especificidad vs F1-Score
   - Identificación del umbral óptimo

### Análisis de Umbrales de Decisión

**Objetivo**: Explorar diferentes umbrales de clasificación (0.5 default vs optimizados).

**Metodología**:
1. Probar umbrales desde 0.10 hasta 0.90
2. Calcular métricas para cada umbral
3. Identificar trade-offs entre sensibilidad y especificidad

**Dataset balanceado**: El umbral default (0.5) suele ser óptimo, pero se exploran alternativas para:
- Maximizar recall (aplicaciones de screening)
- Maximizar F1-Score (balance general)
- Contextos con diferentes costos de FN vs FP

### Interpretación Clínica

- **TP (Verdaderos Positivos)**: Melanomas correctamente detectados
- **TN (Verdaderos Negativos)**: No melanomas correctamente clasificados
- **FP (Falsos Positivos)**: Derivaciones innecesarias (bajo riesgo para paciente)
- **FN (Falsos Negativos)**: Melanomas NO detectados (ALTO RIESGO - puede ser fatal)

### Archivos Generados
- `evaluation_results.json` - Métricas completas en formato JSON
- `roc_curve.png` - Curva ROC
- `precision_recall_curve.png` - Curva Precision-Recall
- `confusion_matrices_comparison.png` - Matrices comparativas
- `threshold_tradeoff.png` - Análisis de umbrales

</details>

<details>
<summary><b>Explainable AI (XAI) - Grad-CAM (Click para expandir)</b></summary>

### ¿Por qué XAI en Diagnóstico Médico?

1. **Confianza clínica**: Los médicos necesitan entender las decisiones del modelo
2. **Validación de aprendizaje**: Verificar que aprende patrones médicamente relevantes
3. **Detección de sesgos**: Identificar enfoque en artefactos irrelevantes
4. **Regulación**: Modelos médicos deben ser interpretables y auditables
5. **Mejora iterativa**: Entender fallos para refinar el modelo

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**Funcionamiento**:
- Genera mapas de calor mostrando qué regiones influyen en la predicción
- Utiliza gradientes para identificar áreas importantes
- Overlay sobre imagen original para interpretación visual

**Colores en Heatmaps**:
- **Rojo/Amarillo**: Influencia POSITIVA (características de melanoma)
- **Azul/Frío**: Sin influencia o influencia negativa

### Casos Analizados

Se generaron visualizaciones para 4 categorías:

#### 1. Verdaderos Positivos (TP)
- Melanomas correctamente detectados
- **Qué buscar**: Enfoque en bordes irregulares, variación de color, asimetría
- **Archivo**: `gradcam_true_positives.png`

#### 2. Falsos Negativos (FN) - CRÍTICO
- Melanomas NO detectados por el modelo
- **Análisis**: ¿Por qué falló? Características atípicas, mala calidad, artefactos
- **Archivo**: `gradcam_false_negatives.png`

#### 3. Falsos Positivos (FP)
- No melanomas clasificados incorrectamente como melanoma
- **Análisis**: Lesiones benignas con características similares
- **Archivo**: `gradcam_false_positives.png`

#### 4. Verdaderos Negativos (TN)
- No melanomas correctamente clasificados
- **Qué buscar**: Bordes regulares, color homogéneo, baja activación
- **Archivo**: `gradcam_true_negatives.png`

### Criterios de Validación Clínica

El modelo es confiable si:
1. Heatmaps en TP coinciden con criterios ABCDE de melanoma
2. FN tienen explicación médica (lesiones tempranas, atípicas)
3. FP no son por artefactos técnicos (pelos, burbujas, marcadores)
4. Modelo NO se enfoca en regiones irrelevantes consistentemente

### Patrones Médicamente Relevantes

**Para MELANOMA** (deberían activar el modelo):
- Asimetría en bordes
- Bordes irregulares
- Variación de color
- Diámetro > 6mm
- Evolución/cambios

**Señales de ALERTA** (sesgos a evitar):
- Enfoque en pelos o burbujas
- Activación en fondo en lugar de lesión
- Atención en marcadores de piel
- Activación fuera de región de interés

### Archivos Generados
- `gradcam_true_positives.png` - Melanomas detectados
- `gradcam_false_negatives.png` - Melanomas perdidos (análisis crítico)
- `gradcam_false_positives.png` - Alarmas incorrectas
- `gradcam_true_negatives.png` - No melanomas correctos

</details>

---
