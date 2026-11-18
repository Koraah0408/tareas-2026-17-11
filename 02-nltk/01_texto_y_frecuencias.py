# Capitulo 1 - Language Processing and Python
# Diapositivas 3-8 y 13: Searching Text, Conjuntos, Frequency Distributions,
# Collocations and Bigrams, Conteos / Longitud de palabra, Generating Random Text with Bigrams.
import nltk
from nltk.book import text1, text2, text4, text8
from nltk import FreqDist, bigrams


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------- Searching Text ----------------
titulo("from nltk.book import *  ->  textos disponibles")
print(text1)
print(text2)

titulo("Concordance: 'monstrous' en Moby Dick")
text1.concordance("monstrous")

titulo("Similar: palabras en contextos parecidos a 'monstrous'")
print("text1 (Moby Dick):")
text1.similar("monstrous")
print("text2 (Sense and Sensibility):")
text2.similar("monstrous")

# ---------------- Conjuntos ----------------
titulo("Conjuntos")
saying = ['After', 'all', 'is', 'said', 'and', 'done',
          'more', 'is', 'said', 'than', 'done']
tokens = set(saying)
print("set(saying) =", tokens)
tokens = sorted(tokens)
print("sorted ->", tokens)
print("tokens[-2:] =", tokens[-2:])

# ---------------- Frequency Distributions ----------------
titulo("Frequency Distributions (vocabulario del texto / numero de repeticiones)")
fdist1 = FreqDist(text1)
print(fdist1)
vocabulary1 = [w for w, _ in fdist1.most_common()]   # en NLTK 3 keys() ya no viene ordenado
print("50 mas frecuentes:", vocabulary1[:50])
print("fdist1['whale'] =", fdist1['whale'])

# ---------------- Collocations and Bigrams ----------------
titulo("Bigrams y Collocations")
print(list(bigrams(['more', 'is', 'said', 'than', 'done'])))
print("\ntext4.collocations():")
text4.collocations()
print("\ntext8.collocations():")
text8.collocations()

# ---------------- Conteos: longitud de palabra ----------------
titulo("Conteos - Longitud de palabra")
print([len(w) for w in text1][:25], "...")
fdist = FreqDist(len(w) for w in text1)
print(fdist)
print("keys:", sorted(fdist.keys()))
print("items (mayor frecuencia primero):", fdist.most_common())
print("fdist.max() =", fdist.max())
print("fdist[3] =", fdist[3])
print("fdist.freq(3) =", fdist.freq(3))

# ---------------- Tabla 1-2: funciones de FreqDist ----------------
titulo("Tabla 1-2: funciones de FreqDist")
fd = FreqDist(text1)
print("fdist['monstrous']      =", fd['monstrous'])
print("fdist.freq('monstrous') =", fd.freq('monstrous'))
print("fdist.N()               =", fd.N())
print("fdist.max()             =", fd.max())
fd['monstrous'] += 1                                 # equivalente a fdist.inc(sample)
print("despues de incrementar  =", fd['monstrous'])
print("tabulate (10 mas comunes):")
FreqDist(dict(fd.most_common(10))).tabulate()
print("fdist1 < fdist2 ? (FreqDist de text1 vs text2):", FreqDist(text2) < FreqDist(text1))
print("(fdist.plot() y fdist.plot(cumulative=True) abren una grafica; ver 'graficas' abajo)")

# ---------------- Generating Random Text with Bigrams ----------------
titulo("Generating Random Text with Bigrams")
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
        'and', 'the', 'earth', '.']
print(list(nltk.bigrams(sent)))


def generate_model(cfdist, word, num=15):
    salida = []
    for i in range(num):
        salida.append(word)
        word = cfdist[word].max()
    print(" ".join(salida))


text = nltk.corpus.genesis.words('english-kjv.txt')
cfd = nltk.ConditionalFreqDist(nltk.bigrams(text))
print("cfd['living'] =", dict(cfd['living']))
generate_model(cfd, 'living')

# ---------------- graficas (opcional) ----------------
if __name__ == "__main__":
    import sys
    if "--graficas" in sys.argv:
        fdist1.plot(50, cumulative=True)
