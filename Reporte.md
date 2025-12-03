# Clasificación Automática de Neumonía en Radiografías de Tórax
## Basada en descriptores tradicionales y modelos de Deep Learning

**Autores:**  
- Laura Sanín Colorado  
- Juan Manuel Sanchez Restrepo  
- Sebastián Palacio Betancur  
- Henrry Uribe Cabrera Ordoñez

---

# 1. Introducción

La identificación temprana y precisa de patologías pulmonares mediante radiografías de tórax es un componente fundamental del diagnóstico clínico. La interpretación manual depende fuertemente de la experiencia del especialista y puede verse limitada en instituciones con alta demanda, escasez de radiólogos o variabilidad interobservador.  

Los sistemas de apoyo al diagnóstico basados en visión por computador ofrecen una alternativa para aumentar la precisión y estandarizar interpretaciones.  

Este proyecto aborda la **clasificación binaria de radiografías** (normales vs neumonía), usando:

1. **Descriptores tradicionales** (HOG, LBP, Haralick, Zernike) + modelos clásicos de ML.  
2. **Modelos profundos preentrenados** (ResNet con Transfer Learning).  

---

## 1.1. Objetivos

**General:**  
- Desarrollar un sistema automatizado capaz de clasificar radiografías de tórax como normales o con neumonía, evaluando la eficacia de enfoques tradicionales y de deep learning.

**Específicos:**  
- Analizar la calidad y características del dataset.  
- Extraer y evaluar descriptores tradicionales de imágenes.  
- Entrenar y comparar clasificadores clásicos y modelos de deep learning.  
- Determinar métricas de desempeño y robustez frente a variabilidad de imágenes.  
- Proponer mejoras y posibles aplicaciones clínicas.

---

# 2. Marco Teórico

## 2.1. Preprocesamiento de Imágenes Médicas

- **Redimensionamiento:** uniformidad de dimensiones.  
- **Escala de grises:** reduce dimensionalidad manteniendo información clínica.  
- **Normalización:** homogeneiza intensidades entre imágenes.  
- **CLAHE:** mejora contraste local evitando amplificación de ruido.  

Estas técnicas permiten consistencia y comparabilidad entre descriptores tradicionales y redes neuronales.

---

## 2.2. Descriptores Tradicionales

- **HOG:** captura gradientes locales, ideal para bordes y estructuras anatómicas.  
- **LBP:** describe microtexturas robustas a cambios de iluminación.  
- **Haralick / GLCM:** atributos de homogeneidad, contraste y correlación.  
- **Momentos de Zernike:** invariantes a rotación, capturan información global de forma.  

---

## 2.3. Métodos de Clasificación

- **SVM**, **Random Forest**, **XGBoost**, **k-NN**, **Regresión Logística**.  
- Todos requieren vectores de características construidos previamente.

---

## 2.4. Deep Learning — Transfer Learning

- **ResNet preentrenada en ImageNet**.  
- Capas convolucionales como extractor de características + capas finales ajustadas a clasificación binaria.  
- *Skip connections* permiten aprender representaciones complejas sin degradación.

---

# 3. Metodología

## 3.0. Dataset

El proyecto utilizó un dataset público de radiografías de tórax, compuesto por imágenes etiquetadas como **normales** o **con neumonía**:

- **Fuente:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- **Número total de imágenes:** 5,856  
  - **Normales:** 1,583  
  - **Neumonía:** 4,273  
- **Formato:** JPEG, resolución variable (principalmente 1024×1024 px)  
- **División de entrenamiento y prueba:**  
  - **Train:** 70%  
  - **Test:** 30%  
- **Notas:** Las imágenes de neumonía incluyen casos bacterianos y virales sin distinción en este estudio.  

**Importancia del dataset:**  
- Tamaño suficiente para entrenar modelos tradicionales y realizar Transfer Learning con CNNs.  
- Diversidad de imágenes que permite evaluar la robustez de los descriptores y la generalización de los modelos.

---

## 3.1. Notebook 01 — Exploratory Analysis & Cleaning

- Carga del dataset, visualización de tamaños, contraste, nitidez y preprocesamiento (gris, redimensionamiento, normalización, CLAHE).  

**Observaciones Clave:**  
- **Contraste y textura:** neumonía → zonas homogéneas/densas; normal → mayor detalle.  
- **Distribución de intensidades:** KDE/CDF muestran diferencias consistentes.  
- **Intensidad promedio y variabilidad:** mayor en neumonía, mayor dispersión entre imágenes afectadas.  

> ![Ejemplo Exploratory Analysis](ruta/de/la/imagen_exploratory.png)

---

## 3.2. Notebook 02 — Feature Extraction

- **26,244 dimensiones de características** extraídas: HOG, LBP, GLCM, Momentos de Hu, filtros Gabor y estadísticas básicas.  

### HOG
- Detecta costillas, columna, contornos pulmonares.  
- Valores normalizados: -7.58 a 10.  

> ![Ejemplo HOG](ruta/de/la/imagen_HOG.png)

### Momentos de Hu y Geometría
- Área: 22,372 px, Perímetro: 4,243  
- Excentricidad: 0.57, Solidez: 0.60, Relación de aspecto: 1.10  

### LBP
- Textura local robusta a iluminación  
- Diferencia tejido sano vs patológico  

> ![Ejemplo LBP](ruta/de/la/imagen_LBP.png)

### GLCM
- 60 dimensiones  
- Contraste: 9.52, Disimilitud: 1.89, Homogeneidad: 0.49  

### Filtros Gabor
- Detectan patrones según frecuencia/orientación  
- Capturan estructuras anatómicas y detalles finos

### Estadísticas básicas
- Media: 0.52, SD: 0.24, Varianza: 0.058  
- Skewness: -0.26, Kurtosis: -0.92, Entropía: -43.78  

**Conclusión:** conjunto de características robusto y diferenciador.

> ![Resumen de extracción de características](ruta/de/la/imagen_resumen.png)

---

## 3.3. Notebook 03 — Classification

- División train/test, estandarización, entrenamiento de clasificadores tradicionales y ResNet Transfer Learning.  
- Evaluación con matriz de confusión, precision, recall, F1, AUC-ROC.  
- Comparación modelos tradicionales vs ResNet.  
- Análisis de importancia de características.

---

# 4. Experimentos y Resultados

## 4.1. Comparación de enfoques

| Modelo / Enfoque      | Accuracy | Precision | Recall  | F1-Score | AUC-ROC |
|-----------------------|---------|-----------|--------|----------|---------|
| ResNet Transfer (DL)  | 83.65%  | 79.88%    | 98.7%  | 88.3%    | 0.945   |
| k-NN (Tradicional)    | 79%     | 74.5%     | 88%    | 80.7%    | 0.873   |
| Random Forest         | 73%     | 71%       | 82%    | 76.5%    | 0.815   |
| XGBoost               | 68%     | 65%       | 78%    | 70%      | 0.78    |
| SVM                   | 64%     | 62%       | 76%    | 68%      | 0.755   |

---

## 4.2. Observaciones

- Modelos tradicionales dependen del preprocesamiento y son sensibles al contraste y ruido.  
- ResNet Transfer Learning: mejor F1-score y generalización; captura patrones complejos sin ingeniería manual.  
- Estrategias frente al desbalance: weighting, oversampling y control de métricas.

---

# 5. Análisis y Discusión

- **Balance de clases:** dataset desbalanceado, mitigado con técnicas de ponderación y sampling.  
- **Comparación de enfoques:** Deep Learning más robusto; tradicionales interpretables y útiles en escenarios de recursos limitados.  
- **Limitaciones:** falta de augmentations fuertes, alta variabilidad, ausencia de validación externa.  
- **Mejoras sugeridas:** fine-tuning completo de ResNet, aumento de dataset, augmentations realistas, ensemble DL + tradicionales, validación cruzada k-fold.  
- **Aplicaciones clínicas:** soporte a radiólogos, estandarización de diagnósticos, reducción de tiempo de análisis, posible integración en PACS hospitalarios.

---

# 6. Conclusiones

- Preprocesamiento es esencial para la homogeneidad y calidad de características.  
- Descriptores tradicionales funcionan, pero con limitaciones frente a variabilidad anatómica y contraste.  
- ResNet Transfer Learning supera métricas clásicas; robusto y eficiente.  
- KDE/CDF y extracción de características confirman diferencias de contraste y textura entre clases.  
- Aplicaciones clínicas: soporte a radiólogos, estandarización de diagnósticos y reducción de tiempo de análisis.

---

# 7. Referencias (APA)

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR.  
- Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI.  
- Haralick, R. M. (1973). *Textural features for image classification*. IEEE TSMC.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.  
- Pizer, S. et al. (1987). *Adaptive histogram equalization and its variations*. CVGIP.

---

# 8. Contribución del Equipo

- **Henrry Uribe Cabrera Ordoñez:** Reporte, Análisis de datos, extracción de características, entrenamiento y evaluación de modelos.  
- **Laura Sanín Colorado:** preprocesamiento de imágenes, análisis exploratorio y documentación de resultados y reporte.  
- **Juan Manuel Sanchez Restrepo:** implementación de clasificadores tradicionales, ajuste de parámetros y generación de gráficos de desempeño, entrenamiento y evalucion del modelo.  
- **Sebastián Palacio Betancur:** implementación de Transfer Learning con ResNet, evaluación de modelos de deep learning y comparación con modelos tradicionales.

> Todos los integrantes participaron activamente en la discusión de resultados, redacción del informe final y revisión del contenido técnico.
