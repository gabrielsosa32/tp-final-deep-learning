# 📰 TP final de Deep Learning — detección de noticias falsas

> *Fake-news classifier using an LSTM over pre-trained Word2Vec (GoogleNews) embeddings — 0.9173 accuracy on 2,093 test articles.*

Trabajo final de la materia de redes neuronales. Tiene dos partes: la principal es un **clasificador de noticias falsas con una LSTM**; la segunda es un ejercicio de **regresión** sobre la popularidad de artículos web.

Todo el desarrollo está en [`TPFinalSosaGabriel.ipynb`](TPFinalSosaGabriel.ipynb).

---

## 1. Clasificación de noticias falsas (LSTM + Word2Vec)

### Datos

`news1.csv`: noticias en inglés con título, texto y etiqueta `FAKE` / `REAL`. Se descartaron las filas con etiquetas mal parseadas y quedaron **6.340 noticias** balanceadas (3.186 FAKE, 3.154 REAL). Split 67/33 (`random_state=42`).

### Modelo

- Tokenización con un vocabulario de 80.000 palabras y secuencias de 80 tokens (`pad_sequences`).
- Capa de **embeddings pre-entrenados Word2Vec de GoogleNews** (300 dimensiones), **congelados** (`trainable=False`).
- `LSTM(128)` → `Dropout(0.5)` → `Dense(1, sigmoid)`.
- Pérdida `binary_crossentropy`, optimizador Adam, 20 épocas, batch de 64.

### Resultados (test: 2.093 noticias)

| Métrica | Valor |
|---|---|
| **Accuracy** | **0,9173** |
| F1 (FAKE / REAL) | 0,92 / 0,91 |
| Precision (FAKE / REAL) | 0,92 / 0,91 |
| Recall (FAKE / REAL) | 0,92 / 0,92 |

Además se probó el modelo sobre 7 noticias nuevas (`news_pred.csv`) y clasificó las 7 correctamente.

**Para tener en cuenta:** el accuracy de entrenamiento llega a 0,998 en las últimas épocas contra 0,917 en test, así que hay sobreajuste; un `EarlyStopping` sobre un conjunto de validación separado es la mejora obvia.

---

## 2. Regresión: popularidad de noticias (Online News Popularity)

`OnlineNewsPopularity.csv`: 39.644 artículos de Mashable con 61 atributos; el objetivo es predecir `shares` (cantidad de veces que se compartió).

- Se eliminaron outliers de `shares` con IQR y 14 variables con correlación > 0,7 entre sí (quedan 44 features).
- Se probaron Random Forest con `GridSearchCV` (200 combinaciones, cv=5), una red densa (512 → 384 → 1) y XGBoost con `GridSearchCV`.
- Resultado en test: **XGBoost R² = 0,1455**, red densa R² = 0,1085.

El R² es bajo: con estas variables, la cantidad de shares es difícil de predecir. Se dejó así, sin maquillar. La última celda (bagging heterogéneo con `util_bagging.py`) quedó sin ejecutar por tiempo de cómputo.

---

## 🚀 Cómo correrlo

```bash
git clone https://github.com/gabrielsosa32/tp-final-deep-learning.git
cd tp-final-deep-learning
pip install -r requirements.txt
jupyter notebook TPFinalSosaGabriel.ipynb
```

El archivo de embeddings `GoogleNews-vectors-negative300.bin` (3,6 GB) **no está en el repo**: el notebook lo descarga automáticamente con `gdown` la primera vez.

---

## 🔧 Tecnologías

Python · TensorFlow / Keras · gensim (Word2Vec) · scikit-learn · XGBoost · pandas · NumPy · Matplotlib · Seaborn · WordCloud
