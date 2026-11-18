# Capitulo 2 - Accessing Text Corpora and Lexical Resources
# Diapositivas 10-12 y 14-25: Brown + ConditionalFreqDist, tabla de corpus, puzzle de palabras,
# nombres de hombre y mujer, diccionario de pronunciacion (CMU), WordNet completo
# (synsets, lemmas, jerarquia, hiponimos/hiperonimos, meronimos/holonimos, entailments,
# antonimos y similitud semantica).
import nltk
from nltk.corpus import brown, wordnet as wn


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------- Brown corpus ----------------
titulo("Brown corpus: verbos modales en 'news'")
news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modals = ['can', 'could', 'may', 'might', 'must', 'will']
print(" ".join(f"{m}: {fdist[m]}" for m in modals))

titulo("ConditionalFreqDist por genero")
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
cfd.tabulate(conditions=genres, samples=modals)

# ---------------- Tabla 2-2: corpus de NLTK ----------------
titulo("Tabla 2-2: algunos corpus incluidos en NLTK (los que se descargaron)")
for nombre in ["brown", "cmudict", "conll2000", "genesis", "gutenberg", "inaugural",
               "movie_reviews", "names", "nps_chat", "shakespeare", "stopwords",
               "treebank", "udhr", "words", "wordnet"]:
    try:
        c = getattr(nltk.corpus, nombre)
        print(f"  {nombre:<14} archivos: {len(c.fileids())}")
    except Exception:
        print(f"  {nombre:<14} (no descargado)")
print("Ejemplo en espanol (cess_esp / udhr):", nltk.corpus.udhr.words('Spanish-Latin1')[:12])

# ---------------- Puzzle de letras ----------------
titulo("Puzzle: palabras con las letras E G I V R V O N L (obligatoria la 'r')")
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
print([w for w in wordlist if len(w) >= 6
       and obligatory in w
       and nltk.FreqDist(w) <= puzzle_letters])

# ---------------- Nombres de hombre y mujer ----------------
titulo("Busqueda de nombres de hombre y a la vez de mujer")
names = nltk.corpus.names
print(names.fileids())
male_names = names.words('male.txt')
female_names = set(names.words('female.txt'))
ambos = [w for w in male_names if w in female_names]
print(len(ambos), "nombres:", ambos[:25], "...")

# ---------------- Diccionario de pronunciacion ----------------
titulo("A Pronouncing Dictionary (CMU)")
entries = nltk.corpus.cmudict.entries()
print("len(entries) =", len(entries))
inicio = next(i for i, (w, _) in enumerate(entries) if w == 'fir')
for entry in entries[inicio:inicio + 8]:
    print(entry)

# ---------------- WordNet ----------------
titulo("2.5 WordNet: synsets y lemmas")
print("wn.synsets('motorcar') =", wn.synsets('motorcar'))
car = wn.synset('car.n.01')
print("lemma_names :", car.lemma_names())
print("definition  :", car.definition())
print("examples    :", car.examples())
print("lemmas      :", car.lemmas())
print("wn.lemma('car.n.01.automobile')        =", wn.lemma('car.n.01.automobile'))
print("wn.lemma('car.n.01.automobile').synset =", wn.lemma('car.n.01.automobile').synset())
print("wn.lemma('car.n.01.automobile').name   =", wn.lemma('car.n.01.automobile').name())

titulo("La palabra 'car' es ambigua: tiene cinco synsets")
print(wn.synsets('car'))
for synset in wn.synsets('car'):
    print(synset.lemma_names())
print("wn.lemmas('car') =", wn.lemmas('car'))

titulo("The WordNet Hierarchy: hyponyms")
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
print("types_of_motorcar[0] =", types_of_motorcar[0])
print(sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())[:40], "...")

titulo("hypernyms")
print("motorcar.hypernyms() =", motorcar.hypernyms())
paths = motorcar.hypernym_paths()
print("len(paths) =", len(paths))
print([synset.name() for synset in paths[0]])
print([synset.name() for synset in paths[1]])
print("root_hypernyms() =", motorcar.root_hypernyms())

titulo("Meronyms, holonyms y entailments")
print("tree part_meronyms      :", wn.synset('tree.n.01').part_meronyms())
print("tree substance_meronyms :", wn.synset('tree.n.01').substance_meronyms())
print("tree member_holonyms    :", wn.synset('tree.n.01').member_holonyms())
print("walk entailments  :", wn.synset('walk.v.01').entailments())
print("eat entailments   :", wn.synset('eat.v.01').entailments())
print("tease entailments :", wn.synset('tease.v.03').entailments())

titulo("Antonymy")
for l in ['supply.n.02.supply', 'rush.v.01.rush', 'horizontal.a.01.horizontal',
          'staccato.r.01.staccato']:
    print(f"{l:<28} ->", wn.lemma(l).antonyms())

titulo("Semantic Similarity")
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
minke = wn.synset('minke_whale.n.01')
tortoise = wn.synset('tortoise.n.01')
novel = wn.synset('novel.n.01')
for nombre, s in [("minke", minke), ("orca", orca), ("tortoise", tortoise), ("novel", novel)]:
    print(f"right.lowest_common_hypernyms({nombre}) =", right.lowest_common_hypernyms(s))
for n in ['baleen_whale.n.01', 'whale.n.02', 'vertebrate.n.01', 'entity.n.01']:
    print(f"{n}.min_depth() =", wn.synset(n).min_depth())
for nombre, s in [("minke", minke), ("orca", orca), ("tortoise", tortoise), ("novel", novel)]:
    print(f"right.path_similarity({nombre}) =", right.path_similarity(s))
