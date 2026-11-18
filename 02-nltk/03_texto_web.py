# Capitulo 3 - Processing Raw Text
# Diapositivas 26-28: 3.1 Accessing Text from the Web and from Disk,
# busqueda de palabras dentro del texto (find / rfind) y paginas HTML.
# Necesita internet.
from urllib.request import urlopen, Request
import nltk
from bs4 import BeautifulSoup   # nltk.clean_html() ya no existe; se usa BeautifulSoup


def titulo(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def descargar(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urlopen(req, timeout=30).read().decode("utf8", errors="replace")


# ---------------- Texto de la web: Crimen y Castigo ----------------
titulo("3.1 Accessing Text from the Web (Crime and Punishment, Project Gutenberg)")
# Se prueba el sitio principal y, si esta caido, un espejo oficial de Gutenberg
for url in ["https://www.gutenberg.org/files/2554/2554-0.txt",
            "https://gutenberg.pglaf.org/2/5/5/2554/2554-0.txt"]:
    try:
        raw = descargar(url)
        break
    except Exception as e:
        print(f"No se pudo con {url} ({e}), probando otro...")
print("url =", url)
print("type(raw) =", type(raw))
print("len(raw)  =", len(raw))
print("raw[:75]  =", repr(raw[:75]))

tokens = nltk.word_tokenize(raw)
print("type(tokens) =", type(tokens))
print("len(tokens)  =", len(tokens))
print("tokens[:10]  =", tokens[:10])

text = nltk.Text(tokens)
print("type(text) =", type(text))
print("text[1024:1062] =", text[1024:1062])

titulo("text.collocations()")
text.collocations()

# ---------------- Busqueda de palabras dentro del texto ----------------
titulo("Busqueda de palabras dentro del texto: find / rfind")
inicio = raw.find("PART I")
fin = raw.rfind("*** END OF THE PROJECT GUTENBERG")   # el libro actual termina con este texto
print("raw.find('PART I')  =", inicio)
print("raw.rfind('*** END OF THE PROJECT GUTENBERG') =", fin)
raw = raw[inicio:fin]
print("despues de recortar, raw.find('PART I') =", raw.find("PART I"))

# ---------------- Paginas HTML ----------------
titulo("Paginas HTML: noticia de la BBC 'Blondes to die out'")
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = descargar(url)
print("html[:60] =", repr(html[:60]))
raw = BeautifulSoup(html, "html.parser").get_text()
tokens = nltk.word_tokenize(raw)
print("tokens[:20] =", tokens[:20])
text = nltk.Text(tokens)
text.concordance("gene")
