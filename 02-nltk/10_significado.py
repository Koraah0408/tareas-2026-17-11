# Capitulo 10 - Analyzing the Meaning of Sentences
# Diapositivas 52-55: 10.1 Natural Language Understanding (preguntas en ingles -> SQL ->
# base de datos city.db), 10.2 Propositional Logic, 10.3 First-Order Logic y la gramatica
# simple-sem.fcfg para traducir oraciones a logica.
import nltk
from nltk import load_parser
from nltk.sem import chat80


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------- 10.1 Querying a Database ----------------
titulo("10.1 Querying a Database: gramatica sql0.fcfg")
nltk.data.show_cfg('grammars/book_grammars/sql0.fcfg')

titulo("Pregunta en ingles -> consulta SQL")
cp = load_parser('grammars/book_grammars/sql0.fcfg')
query = 'What cities are located in China'
trees = list(cp.parse(query.split()))
answer = trees[0].label()['SEM']
answer = [s for s in answer if s]
q = ' '.join(answer)
print("Pregunta:", query)
print("SQL     :", q)

titulo("Ejecutar la consulta sobre city.db")
rows = chat80.sql_query('corpora/city_database/city.db', q)
print(" ".join(r[0] for r in rows))

# ---------------- 10.2 Propositional Logic ----------------
titulo("10.2 Propositional Logic")
read_expr = nltk.sem.Expression.fromstring
print(read_expr('-(P & Q)'))
print(read_expr('P & Q'))
print(read_expr('P | (R -> Q)'))
print(read_expr('P <-> -- P'))

# Probar un argumento: si Sylvania esta al norte de Freedonia, Freedonia no esta al norte de Sylvania
# (en el libro se usa Prover9, que es un programa externo; aqui se usa el probador
#  por tableaux que ya trae NLTK, escribiendo las proposiciones como north(s,f))
SnF = read_expr('north(s,f)')
NotFnS = read_expr('-north(f,s)')
R = read_expr('north(s,f) -> -north(f,s)')
prover = nltk.TableauProver()
print("SnF, SnF -> -FnS  |-  -FnS ?", prover.prove(NotFnS, [SnF, R]))

# Valuacion de un modelo
val = nltk.Valuation([('P', True), ('Q', True), ('R', False)])
dom = set()
g = nltk.Assignment(dom)
m = nltk.Model(dom, val)
for e in ['(P & Q)', '-(P & Q)', '(P & R)', '(P | R)']:
    print(f"  {e:<10} =", m.evaluate(e, g))

# ---------------- 10.3 First-Order Logic ----------------
titulo("10.3 First-Order Logic: Angus walks -> walk(angus), Angus sees Bertie -> see(angus, bertie)")
expr = read_expr('walk(angus)', type_check=True)
print(expr, "  argumento:", expr.argument, " funcion:", expr.function)
expr = read_expr('see(angus, bertie)', type_check=True)
print(expr, "  tipo de la funcion:", expr.function.type)

titulo("Gramatica simple-sem.fcfg: 'Angus gives a bone to every dog'")
parser = load_parser('grammars/book_grammars/simple-sem.fcfg', trace=0)
sentence = 'Angus gives a bone to every dog'
tokens = sentence.split()
for tree in parser.parse(tokens):
    print(tree.label()['SEM'])

titulo("Ejemplo: 'Angus owns a dog. It bit Irene.'")
e = read_expr(r'exists x.(dog(x) & own(angus, x) & bite(x, irene))')
print(e)
print("Variables libres:", e.free())

titulo("Evaluar formulas en un modelo")
v = """
bertie => b
olive => o
cyril => c
boy => {b}
girl => {o}
dog => {c}
walk => {o, c}
see => {(b, o), (c, b), (o, c)}
"""
val = nltk.Valuation.fromstring(v)
dom = val.domain
m = nltk.Model(dom, val)
g = nltk.Assignment(dom)
for f in ['see(olive, cyril)', 'see(cyril, olive)', 'exists x.(girl(x) & walk(x))',
          'all x.(dog(x) -> walk(x))']:
    print(f"  {f:<32} =", m.evaluate(f, g))
