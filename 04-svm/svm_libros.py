# SVM (Support Vector Machine / Maquina de Vectores de Soporte)
# --------------------------------------------------------------
# SVM predice una CATEGORIA (no un numero como el SVR). Ejemplo del pizarron:
#   - Dataset de libros con su descripcion y su genero (datos reales: CMU Book Summary Dataset,
#     Universidad Carnegie Mellon, resumenes de Wikipedia de ~16 mil libros).
#   - Se divide en conjunto de entrenamiento (80%) y de prueba (20%).
#   - El modelo aprende que palabras aparecen en cada genero.
#   - Al final se le da la descripcion de un libro de ejemplo y predice su genero.
#
# Nota: las descripciones del dataset estan en INGLES, asi que las descripciones que le des
# tambien deben estar en ingles.
#
#   python svm_libros.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

AQUI = os.path.dirname(os.path.abspath(__file__))

# ---------------- 1. Cargar el dataset ----------------
df = pd.read_csv(os.path.join(AQUI, "libros_generos.csv"))
print(f"Libros en el dataset: {len(df)}")
print(df.genero.value_counts().to_string(), "\n")
print("Ejemplo de un renglon del dataset:")
print(f"  Titulo: {df.titulo[0]}  |  Genero: {df.genero[0]}")
print(f"  Descripcion: {df.descripcion[0][:200]}...\n")

# ---------------- 2. Dividir en entrenamiento y prueba ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df.descripcion, df.genero, test_size=0.2, random_state=42, stratify=df.genero)
print(f"Entrenamiento: {len(X_train)} libros   |   Prueba: {len(X_test)} libros")

# ---------------- 3. Convertir texto a numeros (TF-IDF) ----------------
# El SVM solo entiende numeros. TF-IDF le da a cada palabra un peso: alto si aparece mucho
# en esa descripcion pero poco en las demas (palabras como 'the' o 'and' pesan casi 0).
vectorizador = TfidfVectorizer(stop_words="english", max_features=5000, sublinear_tf=True)
X_train_num = vectorizador.fit_transform(X_train)
X_test_num = vectorizador.transform(X_test)
print(f"Cada libro se convierte en un vector de {X_train_num.shape[1]} numeros (uno por palabra)\n")

# ---------------- 4. Entrenar el SVM ----------------
modelo = SVC(kernel="linear", C=1.0)
modelo.fit(X_train_num, y_train)
print(f"Modelo entrenado: SVC(kernel='linear', C=1.0), vectores de soporte = {modelo.support_.size}")

# ---------------- 5. Evaluar con el conjunto de prueba ----------------
pred = modelo.predict(X_test_num)
print(f"\nExactitud (accuracy) en prueba: {accuracy_score(y_test, pred) * 100:.1f}%")
print("  (de cada 100 libros que NO vio al entrenar, cuantos clasifico bien)\n")
print("Reporte por genero:")
print(classification_report(y_test, pred, digits=2))

print("Matriz de confusion (renglon = genero real, columna = lo que predijo):")
etiquetas = sorted(df.genero.unique())
cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=etiquetas),
                  index=etiquetas, columns=[e[:8] for e in etiquetas])
print(cm.to_string(), "\n")

# ---------------- 6. Predecir el genero de libros de ejemplo ----------------
ejemplos = [
    "A young girl discovers she is a witch and must use her magic to defeat an evil sorcerer "
    "who wants to rule the kingdom with his army of dragons.",
    "Detective Morgan investigates the murder of a wealthy banker found dead in his locked "
    "study. Every suspect has an alibi, but the clues point to someone close to the family.",
    "The crew of a starship travels to a distant planet where they find an alien "
    "civilization and advanced technology that could save humanity.",
    "Something evil lives in the old house. At night the family hears screams, and one by one "
    "they are haunted by the ghost of a murdered child.",
    "She never expected to fall in love with the charming stranger who moved into town, but "
    "their passion grows despite her family's disapproval of the marriage.",
]
print("Prediccion de libros de ejemplo:")
for desc in ejemplos:
    genero = modelo.predict(vectorizador.transform([desc]))[0]
    print(f"  -> {genero:<16} | {desc[:75]}...")
