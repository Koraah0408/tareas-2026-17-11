# Descarga todos los corpus y modelos de NLTK que usan los demas programas.
# Correr UNA vez antes que todo:  python 00_descargar_datos.py
import nltk

PAQUETES = [
    "book",                      # text1..text9, gutenberg, genesis, inaugural, nps_chat, webtext, treebank, wordnet, names, words...
    "brown", "cmudict", "wordnet", "omw-1.4", "names", "words", "genesis", "movie_reviews",
    "punkt", "punkt_tab",        # tokenizadores de oraciones/palabras
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",   # nltk.pos_tag
    "maxent_ne_chunker", "maxent_ne_chunker_tab",                     # nltk.ne_chunk
    "universal_tagset", "ieer", "conll2000", "shakespeare",
    "book_grammars", "city_database", "sample_grammars",
]

for p in PAQUETES:
    nltk.download(p, quiet=True)
print("Listo, datos de NLTK descargados.")
