# Capitulo 11 - Managing Linguistic Data
# Diapositiva 56: 11.3 Acquiring Data (de la web, de archivos de texto, de hojas de calculo y
# bases de datos) y 11.4 Working with XML (usar XML para estructuras linguisticas).
import csv
import os
import sqlite3
from xml.etree import ElementTree
import nltk


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_cap11")
os.makedirs(CARPETA, exist_ok=True)

# ---------------- Obtaining Data from Word Processor / text files ----------------
titulo("Obtaining Data from Word Processor Files (texto exportado a .txt)")
ruta_txt = os.path.join(CARPETA, "diccionario.txt")
with open(ruta_txt, "w", encoding="utf8") as f:
    f.write("sleep (v.) condition of body and mind\n"
            "walk (v.) progress by lifting and setting down each foot\n"
            "wake (v.) cease to sleep\n")
legal_pos = {'n', 'v.', 'adj', 'v.i'}
pattern = r"^(\w+)\s+\(([\w.]+)\)"
import re
for line in open(ruta_txt, encoding="utf8"):
    m = re.match(pattern, line)
    if m and m.group(2) in legal_pos:
        print(f"  palabra={m.group(1):<6} categoria={m.group(2)}")

# ---------------- Obtaining Data from Spreadsheets and Databases ----------------
titulo("Obtaining Data from Spreadsheets (CSV)")
ruta_csv = os.path.join(CARPETA, "dict.csv")
with open(ruta_csv, "w", newline="", encoding="utf8") as f:
    w = csv.writer(f)
    w.writerow(["sleep", "sli:p", "v.i", "a condition of body and mind ..."])
    w.writerow(["walk", "wo:k", "v.intr", "progress by lifting and setting down each foot ..."])
    w.writerow(["wake", "weik", "intrans", "cease to sleep"])
lexicon = list(csv.reader(open(ruta_csv, encoding="utf8")))
for row in lexicon:
    print(" ", row)

titulo("Encontrar palabras usadas en las definiciones que no estan definidas")
defined_words = {lexeme for (lexeme, _, _, _) in lexicon}
used_words = {w for (_, _, _, defn) in lexicon for w in nltk.word_tokenize(defn) if w.isalpha()}
print(sorted(used_words - defined_words))

titulo("Obtaining Data from Databases (SQLite)")
con = sqlite3.connect(":memory:")
con.execute("CREATE TABLE lexicon (lexeme TEXT, pron TEXT, pos TEXT, defn TEXT)")
con.executemany("INSERT INTO lexicon VALUES (?,?,?,?)", lexicon)
for fila in con.execute("SELECT lexeme, pos FROM lexicon WHERE pos LIKE 'v.%'"):
    print(" ", fila)

# ---------------- 11.4 Working with XML ----------------
titulo("11.4 Working with XML: The Merchant of Venice (corpus shakespeare)")
merchant_file = nltk.data.find('corpora/shakespeare/merchant.xml')
raw = open(merchant_file).read()
print(raw[:163])
merchant = ElementTree.parse(merchant_file)
print("titulo:", merchant.find('TITLE').text)
print("actos :", [act.find('TITLE').text for act in merchant.findall('ACT')])
print("Escena 1, discurso 0, primeras lineas:")
for line in merchant.findall('ACT')[0].findall('SCENE')[0].findall('SPEECH')[0].findall('LINE')[:3]:
    print("   ", line.text)

titulo("Quien habla mas en 'The Merchant of Venice'")
speaker_seq = [s.text for s in merchant.findall('ACT/SCENE/SPEECH/SPEAKER')]
speaker_freq = nltk.FreqDist(speaker_seq)
print(speaker_freq.most_common(5))

titulo("Using XML for Linguistic Structures: escribir un lexico en XML")
lex = ElementTree.Element("lexicon")
for lexeme, pron, pos, defn in lexicon:
    entry = ElementTree.SubElement(lex, "entry", {"pos": pos})
    ElementTree.SubElement(entry, "lexeme").text = lexeme
    ElementTree.SubElement(entry, "pron").text = pron
    ElementTree.SubElement(entry, "defn").text = defn
ElementTree.indent(lex)
ruta_xml = os.path.join(CARPETA, "lexico.xml")
ElementTree.ElementTree(lex).write(ruta_xml, encoding="utf8", xml_declaration=True)
print(open(ruta_xml, encoding="utf8").read())
print("Leyendolo de vuelta:", [e.findtext("lexeme") for e in ElementTree.parse(ruta_xml).getroot()])
