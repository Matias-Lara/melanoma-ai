# Optimización del Pipeline de Entrenamiento

## Resumen

**Problema:** El entrenamiento inicial era muy lento (más de una hora) y consumía demasiada RAM, causando errores de memoria en Google Colab.

**Solución:** Cambiamos de `ImageDataGenerator` a `tf.data` con Mixed Precision Training, y migramos de Google Colab a Kaggle.

**Resultado:** Entrenamiento 6-8 veces más rápido (10-15 minutos) con menor consumo de memoria y sin pérdida de calidad.

---

## El Problema Original

### Enfoque inicial con ImageDataGenerator

```python
# Código ANTIGUO
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    horizontal_flip=True,
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
model.fit(train_generator, epochs=10)
```

### Por qué era intuitivo usar ImageDataGenerator

`ImageDataGenerator` es el método más popular para data augmentation en tutoriales y documentación de Keras. Aparece en la mayoría de ejemplos oficiales y notebooks de Kaggle, lo que hace que parezca la opción estándar.

**Razones por las que lo elegimos inicialmente:**

- Es el enfoque tradicional que aparece en casi todos los tutoriales
- Sintaxis simple y fácil de entender para principiantes
- Funciona sin configuración compleja
- La documentación oficial de Keras lo usa en muchos ejemplos

Parecía la elección correcta porque si todos lo usan, debe funcionar bien. Es más fácil de implementar y depurar.

---

## Los Problemas Reales

### 1. Muy lento

El entrenamiento tomaba mucho tiempo porque `ImageDataGenerator` procesa las imágenes en la CPU en lugar de usar la GPU. Cada vez que el modelo necesita un batch de imágenes, las transformaciones (rotación, flip, cambios de brillo) se aplican una por una en el procesador, lo que crea un cuello de botella.

El modelo pasa la mayor parte del tiempo esperando a que la CPU termine de preparar las imágenes, mientras la GPU está ociosa. Esto es especialmente problemático con datasets grandes.

### 2. Consumo excesivo de RAM

Google Colab tiene recursos limitados de RAM. El problema es que `ImageDataGenerator` necesita cargar todas las imágenes en memoria desde el inicio, y luego crea copias adicionales durante el entrenamiento:

- Carga el dataset completo en memoria
- Mantiene copias para train, validation y test
- Crea nuevas copias cada vez que aplica transformaciones
- No libera memoria eficientemente entre batches

Esto lleva a que el uso de RAM se acumule hasta exceder los límites de Colab, causando que el kernel se reinicie y se pierda el progreso del entrenamiento.

### 3. No aprovecha las GPUs modernas de Kaggle

Kaggle ofrece GPUs T4 x2 que tienen capacidades especiales para acelerar el entrenamiento. Sin embargo, `ImageDataGenerator` no puede aprovechar estas características:

- No hace augmentation en GPU
- No soporta mixed precision (float16)
- No usa las capacidades de Tensor Cores de las T4

Básicamente estábamos usando hardware moderno como si fuera una computadora antigua.

---

## La Solución: tf.data + Mixed Precision

### Por qué nos cambiamos a Kaggle

Además de las optimizaciones de código, migramos de Google Colab a Kaggle por mejores recursos:

- **Más RAM:** Kaggle ofrece más memoria que Colab Free
- **Mejores GPUs:** T4 x2 vs una sola GPU en Colab
- **Más estable:** Menos interrupciones y timeouts
- **Mejor para datasets grandes:** Manejo más eficiente de datos

### Cambio 1: Pipeline tf.data

```python
# Código NUEVO
def augment_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(64).prefetch(2)

model.fit(train_dataset, epochs=7)
```

**Ventajas:**

1. **Augmentation en GPU:** Las transformaciones se hacen directamente en la GPU usando operaciones TensorFlow nativas, mucho más rápido que en CPU.

2. **Prefetching:** Mientras la GPU entrena un batch, la CPU prepara el siguiente. Esto elimina tiempos de espera.

3. **Mejor uso de memoria:** No carga todo en memoria de golpe, procesa por batches y libera memoria inmediatamente.

4. **Aprovecha hardware moderno:** Usa las capacidades completas de las GPUs T4 de Kaggle.

### Cambio 2: Mixed Precision Training

```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

Usa números de 16 bits (float16) para cálculos intermedios en lugar de 32 bits (float32), lo que:

- Reduce el uso de memoria a la mitad
- Hace los cálculos 2-3 veces más rápidos en GPUs T4
- Mantiene la precisión del modelo (no afecta la calidad)

### Cambio 3: Hiperparámetros ajustados

- **Batch size:** 32 → 64 (aprovecha mejor la GPU)
- **Épocas Etapa 1:** 10 → 7 (suficiente con early stopping)
- **Épocas Etapa 2:** 20 → 15 (converge más rápido)

---

## Comparativa

| Aspecto | ImageDataGenerator | tf.data + Mixed Precision |
|---------|-------------------|---------------------------|
| Augmentation | CPU (lento) | GPU (rápido) |
| Tiempo por step | Muy lento | Muy rápido |
| Uso de RAM | Alto (crasheaba) | Moderado (estable) |
| GPU utilization | Bajo | Alto |
| Tiempo total | 75-90 minutos | 10-15 minutos |
| Plataforma | Google Colab | Kaggle |

---

## Conclusión

El cambio de `ImageDataGenerator` a `tf.data` + Mixed Precision, junto con la migración a Kaggle, nos dio:

- Entrenamiento 6-8 veces más rápido
- Menor consumo de RAM
- Sin pérdida de calidad del modelo
- Código más profesional

La combinación de mejor código y mejor hardware transformó un proceso de más de una hora en uno de 10-15 minutos, haciendo el desarrollo mucho más ágil.

---

**Proyecto:** Melanoma Classification with XAI  
**Hardware:** Kaggle (T4 GPU x2)  
**Última actualización:** 15 de noviembre de 2025
