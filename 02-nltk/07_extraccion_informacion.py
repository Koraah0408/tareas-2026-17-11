# Capitulo 7 - Extracting Information from Text
# Diapositivas 44-51: 7.1 Information Extraction, arquitectura (ie_preprocess),
# 7.2 Chunking (NP-chunker con gramatica regex y arbol), Chinking,
# 7.5 Named Entity Recognition y 7.6 Relation Extraction.
#
#   python 07_extraccion_informacion.py          -> solo texto
#   python 07_extraccion_informacion.py --dibujar -> ademas abre la ventana con el arbol (result.draw())
import re
import sys
import nltk


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------- 7.1 Information Extraction ----------------
titulo("7.1 Information Extraction: Which organizations operate in Atlanta?")
locs = [('Omnicom', 'IN', 'New York'),
        ('DDB Needham', 'IN', 'New York'),
        ('Kaplan Thaler Group', 'IN', 'New York'),
        ('BBDO South', 'IN', 'Atlanta'),
        ('Georgia-Pacific', 'IN', 'Atlanta')]
print(f"{'OrgName':<22}LocationName")
for org, _, loc in locs:
    print(f"{org:<22}{loc}")
query = [e1 for (e1, rel, e2) in locs if rel == 'IN' and e2 == 'Atlanta']
print("Respuesta:", query)

# ---------------- Information Extraction Architecture ----------------
titulo("Information Extraction Architecture: ie_preprocess")


def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)                      # 1. segmentacion en oraciones
    sentences = [nltk.word_tokenize(sent) for sent in sentences]  # 2. tokenizacion
    sentences = [nltk.pos_tag(sent) for sent in sentences]        # 3. etiquetado POS
    return sentences


documento = ("BBDO South in Atlanta, which handles corporate advertising for Georgia-Pacific, "
             "will assume additional duties for brands like Angel Soft toilet tissue. "
             "Ken Haldin is a spokesman for Georgia-Pacific in Atlanta.")
for s in ie_preprocess(documento):
    print(s)

# ---------------- 7.2 Chunking ----------------
titulo("7.2 Chunking: NP-chunker con expresion regular")
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
            ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"), ("cat", "NN")]
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
result.pretty_print()   # el arbol dibujado en texto

titulo("Chunking sobre el corpus CoNLL-2000 (evaluacion)")
from nltk.corpus import conll2000
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
cp_np = nltk.RegexpParser(r"NP: {<[CDJNP].*>+}")
print(cp_np.accuracy(test_sents))

# ---------------- Chinking ----------------
titulo("Chinking: utilizado para eliminar elementos")
grammar = r"""
  NP:
    {<.*>+}          # Chunk everything
    }<VBD|IN>+{      # Chink sequences of VBD and IN
  """
cp = nltk.RegexpParser(grammar)
print(cp.parse(sentence))

# ---------------- 7.5 Named Entity Recognition ----------------
titulo("7.5 Named Entity Recognition: nltk.ne_chunk()")
texto = ("The Washington Monument is the most prominent structure in Washington, D.C. "
         "and one of the city's early attractions. It was built in honor of George Washington, "
         "who led the country to independence and then became its first President.")
for sent in ie_preprocess(texto):
    arbol = nltk.ne_chunk(sent)
    for sub in arbol.subtrees():
        if sub.label() != 'S':
            print(f"  {sub.label():<12} {' '.join(w for w, t in sub.leaves())}")

print("\nCon binary=True (solo marca 'NE' sin tipo):")
sent = nltk.corpus.treebank.tagged_sents()[22]
print(" ".join(w for w, t in sent))
for sub in nltk.ne_chunk(sent, binary=True).subtrees(lambda t: t.label() == 'NE'):
    print(f"  NE  {' '.join(w for w, t in sub.leaves())}")

# ---------------- 7.6 Relation Extraction ----------------
titulo("7.6 Relation Extraction: (ORG, 'in', LOC) en el corpus IEER")
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
n = 0
for fileid in nltk.corpus.ieer.fileids():
    for doc in nltk.corpus.ieer.parsed_docs(fileid):
        for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=IN):
            print(" ", nltk.sem.rtuple(rel))
            n += 1
print(f"{n} relaciones encontradas")

if "--dibujar" in sys.argv:
    result.draw()
