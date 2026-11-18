# Capitulo 6 - Learning to Classify Text
# Diapositivas 35-43: 6.1 Supervised Classification, Gender Identification, rastrear errores,
# clasificar documentos, clasificar considerando el contexto (palabra anterior),
# 6.3 Evaluation (Test Set, Accuracy, Precision/Recall/F, Confusion Matrices,
# Cross-Validation) y 6.4 Decision Trees.
import random
import nltk
from nltk.corpus import names, brown, movie_reviews

random.seed(0)  # para que los resultados se puedan repetir


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------- Gender Identification ----------------
titulo("Gender Identification: extractor de rasgos (ultima letra del nombre)")


def gender_features(word):
    return {'last_letter': word[-1]}


print("gender_features('Shrek') =", gender_features('Shrek'))

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

featuresets = [(gender_features(n), g) for (n, g) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("classify('Neo')     =", classifier.classify(gender_features('Neo')))
print("classify('Trinity') =", classifier.classify(gender_features('Trinity')))

titulo("Evaluar con datos no vistos")
print("accuracy =", nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

# ---------------- Rastrear errores individualmente ----------------
titulo("Rastrear errores individualmente (devtest)")
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]
train_set = [(gender_features(n), g) for (n, g) in train_names]
devtest_set = [(gender_features(n), g) for (n, g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("accuracy devtest =", nltk.classify.accuracy(classifier, devtest_set))

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((tag, guess, name))
for (tag, guess, name) in sorted(errors)[:15]:
    print('correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name))
print(f"... ({len(errors)} errores en total)")

titulo("Mejorando el extractor con los errores (sufijos de 1 y 2 letras)")


def gender_features2(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}


train_set = [(gender_features2(n), g) for (n, g) in train_names]
devtest_set = [(gender_features2(n), g) for (n, g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("accuracy devtest =", nltk.classify.accuracy(classifier, devtest_set))

# ---------------- Clasificar documentos ----------------
titulo("Tambien se puede utilizar para clasificar documentos (movie reviews)")
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = [w for w, _ in all_words.most_common(2000)]


def document_features(document):
    document_words = set(document)
    return {f'contains({word})': (word in document_words) for word in word_features}


doc_sets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = doc_sets[100:], doc_sets[:100]
doc_classifier = nltk.NaiveBayesClassifier.train(train_set)
print("accuracy =", nltk.classify.accuracy(doc_classifier, test_set))
doc_classifier.show_most_informative_features(5)

# ---------------- Clasificar con contexto ----------------
titulo("Al clasificar se puede considerar el contexto (palabra anterior)")


def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i - 1]
    return features


print("pos_features(brown.sents()[0], 8) =", pos_features(brown.sents()[0], 8))
tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent, i), tag))
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("accuracy =", nltk.classify.accuracy(classifier, test_set))

# ---------------- 6.3 Evaluation ----------------
titulo("6.3 Evaluation: Accuracy, Precision, Recall y F-Measure")
train_set = [(gender_features(n), g) for (n, g) in train_names]
test_set = [(gender_features(n), g) for (n, g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
gold = [g for (_, g) in test_set]
pred = [classifier.classify(f) for (f, _) in test_set]
print("Accuracy =", nltk.classify.accuracy(classifier, test_set))
for clase in ['female', 'male']:
    TP = sum(1 for g, p in zip(gold, pred) if p == clase and g == clase)
    FP = sum(1 for g, p in zip(gold, pred) if p == clase and g != clase)
    FN = sum(1 for g, p in zip(gold, pred) if p != clase and g == clase)
    TN = sum(1 for g, p in zip(gold, pred) if p != clase and g != clase)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f = 2 * precision * recall / (precision + recall)
    print(f"  clase '{clase}': TP={TP} TN={TN} FP={FP} FN={FN}  "
          f"Precision={precision:.3f}  Recall={recall:.3f}  F-Measure={f:.3f}")

# ---------------- Confusion Matrices ----------------
titulo("4. Confusion Matrices (tagger combinado del capitulo 5)")
brown_tagged_sents = brown.tagged_sents(categories='news')
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)
t2 = nltk.BigramTagger(brown_tagged_sents, backoff=t1)


def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]


def apply_tagger(tagger, corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]


gold = tag_list(brown.tagged_sents(categories='editorial'))
test = tag_list(apply_tagger(t2, brown.tagged_sents(categories='editorial')))
cm = nltk.ConfusionMatrix(gold, test)
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

# ---------------- Cross-Validation ----------------
titulo("5. Cross-Validation (10 particiones, clasificador de genero)")
featuresets = [(gender_features2(n), g) for (n, g) in labeled_names]
k = 10
tam = len(featuresets) // k
scores = []
for i in range(k):
    test_fold = featuresets[i * tam:(i + 1) * tam]
    train_fold = featuresets[:i * tam] + featuresets[(i + 1) * tam:]
    c = nltk.NaiveBayesClassifier.train(train_fold)
    scores.append(nltk.classify.accuracy(c, test_fold))
print("accuracy por particion:", [round(s, 3) for s in scores])
print("promedio:", round(sum(scores) / k, 3))

# ---------------- 6.4 Decision Trees y comparacion ----------------
titulo("6.4 Decision Trees vs Naive Bayes vs Maximum Entropy (genero)")
train_set = [(gender_features2(n), g) for (n, g) in train_names[:2000]]
test_set = [(gender_features2(n), g) for (n, g) in test_names]
tree = nltk.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
print("Decision tree   accuracy =", nltk.classify.accuracy(tree, test_set))
print("\n".join(tree.pretty_format(depth=2).splitlines()[:12]), "\n... (arbol recortado)")
nb = nltk.NaiveBayesClassifier.train(train_set)
print("Naive Bayes     accuracy =", nltk.classify.accuracy(nb, test_set))
me = nltk.MaxentClassifier.train(train_set, max_iter=10, trace=0)
print("Maximum Entropy accuracy =", nltk.classify.accuracy(me, test_set))
