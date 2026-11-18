# Menu para correr cada parte de la presentacion myNLTK.
#   python menu.py
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
OPCIONES = [
    ("00_descargar_datos.py", "Descargar datos de NLTK (correr primero, una sola vez)"),
    ("01_texto_y_frecuencias.py", "Cap 1: concordance, similar, FreqDist, bigrams, collocations, texto aleatorio"),
    ("02_corpus_y_wordnet.py", "Cap 2: Brown, corpus, puzzle, nombres, CMU, WordNet y similitud"),
    ("03_texto_web.py", "Cap 3: texto de la web y HTML (necesita internet)"),
    ("05_etiquetado.py", "Cap 5: pos_tag, diccionarios, regexp/unigram/bigram taggers"),
    ("06_clasificacion.py", "Cap 6: clasificar genero/documentos, evaluacion, matriz de confusion, arboles"),
    ("07_extraccion_informacion.py", "Cap 7: extraccion de informacion, chunking, chinking, NER, relaciones"),
    ("09_sistema_dialogo.py", "Sistemas de dialogo: chatbot tutor (escribe 'salir' para terminar)"),
    ("10_significado.py", "Cap 10: pregunta -> SQL, logica proposicional y de primer orden"),
    ("11_datos_linguisticos.py", "Cap 11: datos de archivos, CSV, base de datos y XML"),
    ("nltk_interactivo.py", "VENTANA INTERACTIVA: buscar, frecuencias, WordNet, etiquetar, genero y chatbot"),
]

# Si faltan los datos de NLTK, se descargan solos antes de mostrar el menu
try:
    import nltk
    for recurso in ["corpora/gutenberg", "corpora/brown", "corpora/wordnet", "taggers/averaged_perceptron_tagger_eng"]:
        nltk.data.find(recurso)
except LookupError:
    print("Faltan datos de NLTK, descargandolos (solo la primera vez, tarda unos minutos)...")
    subprocess.run([sys.executable, os.path.join(AQUI, "00_descargar_datos.py")])

while True:
    print("\n===== myNLTK =====")
    for i, (_, desc) in enumerate(OPCIONES):
        print(f" {i}) {desc}")
    print(" t) Todo (del 1 al 11, sin el chatbot)")
    print(" q) Salir")
    op = input("Opcion: ").strip().lower()
    if op == "q":
        break
    if op == "t":
        elegidos = [a for a, _ in OPCIONES[1:] if a not in ("09_sistema_dialogo.py", "nltk_interactivo.py")]
    elif op.isdigit() and int(op) < len(OPCIONES):
        elegidos = [OPCIONES[int(op)][0]]
    else:
        print("Opcion no valida")
        continue
    for archivo in elegidos:
        print(f"\n>>> python {archivo}")
        subprocess.run([sys.executable, os.path.join(AQUI, archivo)])
