# Tareas de Inteligencia Artificial

Programas en Python de varias prácticas de IA y procesamiento de lenguaje natural.
Casi todas tienen una versión simple (imprime resultados en la terminal) y una **versión interactiva**
con ventana: al pasar el mouse sobre cualquier control, un cuadro amarillo explica qué hace.

| Carpeta | Tema | Cómo correrlo |
|---|---|---|
| `01-red-neuronal` | Red neuronal con TensorFlow/Keras que aprende a convertir kilómetros a millas (perceptrón de 1 neurona o multicapa 3+3+1) | `python red_neuronal_km_millas.py` · `python red_neuronal_interactiva.py` |
| `02-nltk` | Los temas de la presentación de NLTK: concordance, FreqDist, bigramas, WordNet, etiquetado, clasificación, chunking, entidades, SQL con gramáticas, lógica y XML | `python 00_descargar_datos.py` (una vez) · `python menu.py` · `python nltk_interactivo.py` |
| `03-svr` | SVR (regresión): entrena con el S&P 500 de 2010–2016 y predice 2017–2018 | `python svr_sp500.py` · `python svr_interactivo.py` |
| `04-svm` | SVM (clasificación): predice el género de un libro a partir de su descripción | `python svm_libros.py` · `python svm_interactivo.py` |
| `05-knn-irradiancia` | KNN (k = 3) en Excel para predecir la irradiancia global usando 3 atributos | Abrir el `.xlsx` |
| `06-ngramas-cuentos` | N-gramas de palabras y de caracteres en cuentos: conteos, probabilidades, texto generado y perplejidad | `python ngramas_cuentos.py` · `python ngramas_interactivo.py` |

Cada programa se corre **desde su propia carpeta** (por ejemplo `cd 03-svr`).

## Requisitos

- Python 3.10 o más reciente
- Librerías:

```bash
pip install tensorflow nltk scikit-learn pandas numpy matplotlib openpyxl beautifulsoup4
```

- Para la carpeta `02-nltk`, descargar una vez los datos de NLTK: `python 02-nltk/00_descargar_datos.py`

## Datos usados

- **S&P 500 mensual (2010–2018):** [datasets/s-and-p-500](https://github.com/datasets/s-and-p-500), basado en los datos públicos de Robert Shiller.
- **Libros y géneros:** subconjunto del [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html) (David Bamman, Carnegie Mellon University), con licencia Creative Commons Attribution-ShareAlike.
- **Textos de NLTK:** corpus incluidos en [NLTK](https://www.nltk.org/data.html) (Gutenberg, Brown, WordNet, etc.).
- **Cuentos:** escritos para esta práctica.
- **N-gramas:** basado en el capítulo 4 de *Speech and Language Processing* (Jurafsky y Martin).
