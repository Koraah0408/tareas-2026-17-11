# N-gramas en cuentos
# -------------------
# Basado en el capitulo 4 "Language Modeling with N-grams" (Jurafsky & Martin).
#
# Un N-grama es una secuencia de N palabras seguidas:
#   unigrama (1): "el", "perro", "corre"
#   bigrama  (2): "el perro", "perro corre"
#   trigrama (3): "el perro corre"
#
# El programa:
#   1. Lee todos los cuentos (.txt) de la carpeta "cuentos".
#   2. Cuenta cuantos unigramas, bigramas y trigramas tiene cada cuento (de palabras y de caracteres).
#   3. Muestra los n-gramas mas repetidos.
#   4. Calcula probabilidades de bigramas  P(palabra | anterior) = C(anterior palabra) / C(anterior)
#      y la probabilidad de oraciones completas (con y sin suavizado de Laplace).
#   5. Genera texto al azar con bigramas y trigramas.
#   6. Calcula la perplejidad (entrena con 3 cuentos y prueba con el otro).
#   7. Guarda todo en "resultados_ngramas.xlsx".
#
# Para agregar mas cuentos: pon otro .txt en la carpeta "cuentos" (primera linea = titulo).
#
#   python ngramas_cuentos.py
import os
import re
import math
import random
from collections import Counter

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(AQUI, "cuentos")
INICIO, FIN = "<s>", "</s>"          # marcas de inicio y fin de oracion (como en el libro)


def titulo(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---------------- 1. Leer y preparar los cuentos ----------------
def oraciones_de(texto):
    """Divide el texto en oraciones y cada oracion en palabras (minusculas, sin signos)."""
    oraciones = []
    for o in re.split(r"[.!?:;]+", texto.lower()):
        palabras = re.findall(r"[a-záéíóúüñ]+", o)
        if palabras:
            oraciones.append(palabras)
    return oraciones


def ngramas(oraciones, n):
    """Lista de n-gramas. Para n >= 2 se agregan <s> al inicio y </s> al final de cada oracion."""
    lista = []
    for palabras in oraciones:
        if n == 1:
            tokens = palabras
        else:
            tokens = [INICIO] * (n - 1) + palabras + [FIN]
        for i in range(len(tokens) - n + 1):
            lista.append(tuple(tokens[i:i + n]))
    return lista


def ngramas_caracteres(oraciones, n):
    """N-gramas de LETRAS dentro de cada palabra. Ej. 'perro' con n=2 -> pe, er, rr, ro."""
    lista = []
    for palabras in oraciones:
        for p in palabras:
            for i in range(len(p) - n + 1):
                lista.append(p[i:i + n])
    return lista


cuentos = {}
for archivo in sorted(os.listdir(CARPETA)):
    if archivo.endswith(".txt"):
        with open(os.path.join(CARPETA, archivo), encoding="utf8") as f:
            nombre = f.readline().strip()
            cuentos[nombre] = oraciones_de(f.read())
todas = [o for ors in cuentos.values() for o in ors]
print(f"Cuentos leidos: {len(cuentos)}  ->  {', '.join(cuentos)}")

# ---------------- 2. Ejemplo de la clase ----------------
titulo("0. EJEMPLO DE LA CLASE: 'El perro corre rapido' (cada grupo separado por punto)")
clase = oraciones_de("El perro corre rápido")
for n, nombre in [(1, "Unigramas (cada grupo contiene 1 palabra)"),
                  (2, "Bigramas (cada grupo contiene 2 palabras)"),
                  (3, "Trigramas (cada grupo contiene 3 palabras)")]:
    grupos = ngramas(clase, n) if n == 1 else [g for g in ngramas(clase, n) if INICIO not in g and FIN not in g]
    print(f"  {nombre}: " + ". ".join(" ".join(g) for g in grupos) + ".")
print("\n  Tambien se puede con CARACTERES (letras seguidas) de la palabra 'perro':")
for n in (1, 2, 3):
    print(f"    {n} letra(s): " + ". ".join(ngramas_caracteres([["perro"]], n)) + ".")

titulo("1. EJEMPLO: como se parte una oracion de los cuentos en n-gramas")
ejemplo = todas[0]
print("Oracion:", " ".join(ejemplo))
for n, nombre in [(1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas")]:
    grupos = ngramas([ejemplo], n)
    print(f"\n{nombre} ({len(grupos)}):")
    print("  " + "  |  ".join(" ".join(g) for g in grupos))

# ---------------- 3. Conteo por cuento ----------------
titulo("2. CONTEO DE N-GRAMAS POR CUENTO")
filas = []
for nombre, ors in list(cuentos.items()) + [("TODOS LOS CUENTOS", todas)]:
    fila = {"Cuento": nombre, "Oraciones": len(ors),
            "Palabras": sum(len(o) for o in ors),
            "Vocabulario (palabras distintas)": len({p for o in ors for p in o})}
    for n, nom in [(1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas")]:
        g = ngramas(ors, n)
        fila[f"{nom} (total)"] = len(g)
        fila[f"{nom} (distintos)"] = len(set(g))
    filas.append(fila)
resumen = pd.DataFrame(filas)
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(resumen.to_string(index=False))
print("\n(total = cuantos hay contando repetidos; distintos = cuantos diferentes hay)")

# ---------------- 4. N-gramas mas frecuentes ----------------
titulo("3. LOS N-GRAMAS MAS REPETIDOS (en todos los cuentos)")
conteos = {n: Counter(ngramas(todas, n)) for n in (1, 2, 3)}
for n, nom in [(1, "UNIGRAMAS"), (2, "BIGRAMAS"), (3, "TRIGRAMAS")]:
    print(f"\n{nom}:")
    for g, c in conteos[n].most_common(10):
        print(f"  {' '.join(g):<32} {c:>4} veces")

# ---------------- 4b. N-gramas de caracteres ----------------
titulo("3b. N-GRAMAS DE CARACTERES (grupos de letras seguidas dentro de cada palabra)")
filas_car = []
for nombre, ors in list(cuentos.items()) + [("TODOS LOS CUENTOS", todas)]:
    fila = {"Cuento": nombre}
    for n in (1, 2, 3):
        g = ngramas_caracteres(ors, n)
        fila[f"{n} letra (total)" if n == 1 else f"{n} letras (total)"] = len(g)
        fila[f"{n} letra (distintos)" if n == 1 else f"{n} letras (distintos)"] = len(set(g))
    filas_car.append(fila)
resumen_car = pd.DataFrame(filas_car)
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(resumen_car.to_string(index=False))
conteos_car = {n: Counter(ngramas_caracteres(todas, n)) for n in (1, 2, 3)}
for n in (1, 2, 3):
    print(f"\n  {n} letra(s) mas repetidas: " +
          ",  ".join(f"'{g}' ({c})" for g, c in conteos_car[n].most_common(10)))

# ---------------- 5. Probabilidades de bigramas ----------------
titulo("4. PROBABILIDAD DE BIGRAMAS  P(palabra | anterior) = C(anterior palabra) / C(anterior)")
uni = Counter()                                  # conteo de cada palabra como "anterior" (incluye <s>)
for o in todas:
    uni.update([INICIO] + o)
big = conteos[2]
V = len({p for o in todas for p in o}) + 1       # vocabulario + </s>


def p_big(w, prev):
    return big[(prev, w)] / uni[prev] if uni[prev] else 0.0


def p_laplace(w, prev):
    return (big[(prev, w)] + 1) / (uni[prev] + V)


for prev, w in [(INICIO, "había"), ("había", "una"), ("una", "vez"), ("el", "zorro"),
                ("el", "robot"), ("la", "luna"), ("la", "niña"), ("del", "faro")]:
    print(f"  P({w:<6}| {prev:<6}) = C({prev} {w}) / C({prev}) = {big[(prev, w)]:>2} / {uni[prev]:<3} = {p_big(w, prev):.3f}")

print("\nProbabilidad de oraciones completas (se multiplican los bigramas):")


def prob_oracion(frase, funcion):
    tokens = [INICIO] + frase.split() + [FIN]
    log_p = 0.0
    for prev, w in zip(tokens, tokens[1:]):
        p = funcion(w, prev)
        if p == 0:
            return 0.0
        log_p += math.log(p)                     # se suman logaritmos para no tener numeros diminutos
    return math.exp(log_p)


for frase in ["había una vez un zorro", "el robot cerró los ojos", "el zorro cerró los ojos"]:
    print(f"  '{frase}'")
    print(f"      sin suavizado (MLE): {prob_oracion(frase, p_big):.3e}")
    print(f"      con Laplace (add-1): {prob_oracion(frase, p_laplace):.3e}")
print("  (notacion cientifica: 1.653e-05 = 0.00001653)")
print("  -> 'el zorro cerró' nunca aparece en los cuentos: sin suavizado da 0;")
print("     con Laplace (sumar 1 a todos los conteos) ya no es 0. Con Laplace todas bajan,")
print("     porque parte de la probabilidad se reparte entre los bigramas que nunca se vieron.")

# ---------------- 6. Generar texto al azar ----------------
titulo("5. TEXTO GENERADO AL AZAR (como la Figura 4.3 del libro)")
random.seed(7)
sig2, sig3 = {}, {}
for (a, b), c in conteos[2].items():
    sig2.setdefault(a, Counter())[b] += c
for (a, b, c3), c in conteos[3].items():
    sig3.setdefault((a, b), Counter())[c3] += c


def elegir(contador):
    palabras, pesos = zip(*contador.items())
    return random.choices(palabras, weights=pesos)[0]


for k in range(3):
    actual, frase = INICIO, []
    while len(frase) < 25:
        actual = elegir(sig2[actual])
        if actual == FIN:
            break
        frase.append(actual)
    print(f"  Bigramas  {k + 1}: {' '.join(frase)}")
for k in range(3):
    a, b, frase = INICIO, INICIO, []
    while len(frase) < 25:
        c = elegir(sig3[(a, b)])
        if c == FIN:
            break
        frase.append(c)
        a, b = b, c
    print(f"  Trigramas {k + 1}: {' '.join(frase)}")
print("  (con trigramas el texto suena mas parecido a los cuentos originales)")

# ---------------- 7. Perplejidad ----------------
titulo("6. PERPLEJIDAD: entrenar con 3 cuentos y probar con el otro (bigramas + Laplace)")
nombres = list(cuentos)
filas_pp = []
for prueba in nombres:
    entrena = [o for n_, ors in cuentos.items() if n_ != prueba for o in ors]
    u, b = Counter(), Counter(ngramas(entrena, 2))
    for o in entrena:
        u.update([INICIO] + o)
    voc = {p for o in entrena for p in o} | {p for o in cuentos[prueba] for p in o}
    Vp = len(voc) + 1
    log_p, N = 0.0, 0
    for o in cuentos[prueba]:
        tokens = [INICIO] + o + [FIN]
        for prev, w in zip(tokens, tokens[1:]):
            log_p += math.log((b[(prev, w)] + 1) / (u[prev] + Vp))
            N += 1
    pp = math.exp(-log_p / N)
    filas_pp.append({"Cuento de prueba": prueba, "Palabras evaluadas": N, "Perplejidad": round(pp, 1)})
    print(f"  Prueba con '{prueba}': perplejidad = {pp:.1f}")
print("  (perplejidad mas baja = el modelo predice mejor ese cuento)")

# ---------------- 8. Guardar en Excel ----------------
salida = os.path.join(AQUI, "resultados_ngramas.xlsx")


def tabla(n):
    filas = []
    por_cuento = {nom: Counter(ngramas(ors, n)) for nom, ors in cuentos.items()}
    for g, c in conteos[n].most_common():
        fila = {"N-grama": " ".join(g), "Total": c}
        for nom, cnt in por_cuento.items():
            fila[nom] = cnt[g]
        filas.append(fila)
    return pd.DataFrame(filas)


probs = pd.DataFrame([{"Anterior": a, "Palabra": w, "C(anterior palabra)": c, "C(anterior)": uni[a],
                       "P(palabra|anterior)": round(c / uni[a], 4),
                       "P Laplace": round((c + 1) / (uni[a] + V), 4)}
                      for (a, w), c in conteos[2].most_common()])
with pd.ExcelWriter(salida, engine="openpyxl") as xl:
    resumen.to_excel(xl, sheet_name="Resumen", index=False)
    tabla(1).to_excel(xl, sheet_name="Unigramas", index=False)
    tabla(2).to_excel(xl, sheet_name="Bigramas", index=False)
    tabla(3).to_excel(xl, sheet_name="Trigramas", index=False)
    probs.to_excel(xl, sheet_name="Probabilidad bigramas", index=False)
    resumen_car.to_excel(xl, sheet_name="Caracteres - resumen", index=False)
    for n in (2, 3):
        por_c = {nom: Counter(ngramas_caracteres(ors, n)) for nom, ors in cuentos.items()}
        pd.DataFrame([{"Letras": g, "Total": c, **{nom: cnt[g] for nom, cnt in por_c.items()}}
                      for g, c in conteos_car[n].most_common()]).to_excel(
            xl, sheet_name=f"Caracteres de {n}", index=False)
    pd.DataFrame(filas_pp).to_excel(xl, sheet_name="Perplejidad", index=False)
    for hoja in xl.book.worksheets:
        for col in hoja.columns:
            hoja.column_dimensions[col[0].column_letter].width = max(12, min(40, max(len(str(c.value or "")) for c in col) + 2))
print(f"\nResultados guardados en: {salida}")
