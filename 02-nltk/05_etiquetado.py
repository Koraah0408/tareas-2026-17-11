# Capitulo 5 - Categorizing and Tagging Words
# Diapositivas 29-34: 5.1 Using a Tagger, Representing Tagged Tokens, tagset simplificado,
# Defining Dictionaries, Complex Keys and Values, Regular Expression Tagger,
# 5.5 N-Gram Tagging (Unigram) y Combining Taggers (backoff).
import nltk
from collections import defaultdict
from nltk.corpus import brown


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------- 5.1 Using a Tagger ----------------
titulo("5.1 Using a Tagger")
text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))

titulo("Representing Tagged Tokens")
tagged_token = nltk.tag.str2tuple('fly/NN')
print("tagged_token    =", tagged_token)
print("tagged_token[0] =", tagged_token[0])
print("tagged_token[1] =", tagged_token[1])

# ---------------- Tagset simplificado ----------------
titulo("Tagset simplificado (universal) en el Brown corpus")
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
for tag, n in tag_fd.most_common():
    print(f"  {tag:<5} {n}")

# ---------------- Defining Dictionaries ----------------
titulo("Defining Dictionaries")
pos = {}
print("pos =", pos)
pos['colorless'] = 'ADJ'
print("pos =", pos)
pos['ideas'] = 'N'
pos['sleep'] = 'V'
pos['furiously'] = 'ADV'
print("pos =", pos)
print("pos['ideas']     =", pos['ideas'])
print("pos['colorless'] =", pos['colorless'])

# ---------------- Complex Keys and Values ----------------
titulo("Complex Keys and Values")
pos = defaultdict(lambda: defaultdict(int))
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1, w2)][t2] += 1
print("pos[('DET', 'right')] =", dict(pos[('DET', 'right')]))

# ---------------- The Regular Expression Tagger ----------------
titulo("The Regular Expression Tagger")
patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                     # nouns (default)
]
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
regexp_tagger = nltk.RegexpTagger(patterns)
print(regexp_tagger.tag(brown_sents[3]))
print("evaluate =", regexp_tagger.accuracy(brown_tagged_sents))

# ---------------- 5.5 N-Gram Tagging: Unigram ----------------
titulo("5.5 N-Gram Tagging - Unigram Tagging")
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]))
print("evaluate =", unigram_tagger.accuracy(brown_tagged_sents))

# ---------------- Combining Taggers ----------------
titulo("Combining Taggers (bigram -> unigram -> default)")
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print("t2.evaluate(test_sents) =", t2.accuracy(test_sents))
