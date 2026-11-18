# N-gramas interactivo (cuentos)
#   python ngramas_interactivo.py
#
# Pestanas:
#   1. Contar n-gramas ........ eliges cuento, N y si son de palabras o de caracteres, y ves la tabla de n-gramas con cuantas veces aparecen.
#   2. Partir una frase ....... escribes una frase y la ves partida en unigramas, bigramas, trigramas...
#   3. Probabilidad ........... escribes una frase y ves la probabilidad de cada bigrama y de toda la frase.
#   4. Predecir y generar ..... el modelo adivina la siguiente palabra y genera texto al azar.
# Pasa el mouse sobre cualquier cosa y el cuadro amarillo de abajo te explica que es.
import os
import re
import math
import random
import tkinter as tk
from tkinter import ttk
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.join(AQUI, "cuentos")
INICIO, FIN = "<s>", "</s>"
TODOS = "Todos los cuentos"
NOMBRES_N = {1: "Unigramas", 2: "Bigramas", 3: "Trigramas", 4: "4-gramas"}


# ---------------- logica de n-gramas ----------------
def oraciones_de(texto):
    oraciones = []
    for o in re.split(r"[.!?:;]+", texto.lower()):
        palabras = re.findall(r"[a-záéíóúüñ]+", o)
        if palabras:
            oraciones.append(palabras)
    return oraciones


def ngramas(oraciones, n, marcas=True):
    lista = []
    for palabras in oraciones:
        tokens = palabras if (n == 1 or not marcas) else [INICIO] * (n - 1) + palabras + [FIN]
        for i in range(len(tokens) - n + 1):
            lista.append(tuple(tokens[i:i + n]))
    return lista


def ngramas_caracteres(oraciones, n):
    """N-gramas de LETRAS dentro de cada palabra. Ej. 'perro' con n=2 -> pe, er, rr, ro."""
    return [p[i:i + n] for palabras in oraciones for p in palabras for i in range(len(p) - n + 1)]


def a_texto(g):
    return g if isinstance(g, str) else " ".join(g)


cuentos = {}
for archivo in sorted(os.listdir(CARPETA)):
    if archivo.endswith(".txt"):
        with open(os.path.join(CARPETA, archivo), encoding="utf8") as f:
            nombre = f.readline().strip()
            cuentos[nombre] = oraciones_de(f.read())
cuentos[TODOS] = [o for ors in list(cuentos.values()) for o in ors]

AYUDA = {
    "inicio": "QUE ES UN N-GRAMA?  Un grupo de N palabras seguidas. 'el perro corre' tiene 3 unigramas "
              "(el | perro | corre), 2 bigramas (el perro | perro corre) y 1 trigrama (el perro corre). "
              "Contandolos en los cuentos, la computadora aprende que palabras suelen ir juntas. "
              "Pasa el mouse sobre cualquier cosa para ver que es.",
    "tipo": "PALABRAS o CARACTERES: los n-gramas pueden ser grupos de palabras ('el perro', 'perro corre') "
            "o grupos de letras seguidas dentro de cada palabra ('perro' -> pe | er | rr | ro).",
    "cuento": "CUENTO: de que cuento se cuentan los n-gramas. 'Todos los cuentos' junta los 4. "
              "Puedes agregar mas cuentos poniendo un .txt en la carpeta 'cuentos' (primera linea = titulo).",
    "n": "N: cuantas palabras tiene cada grupo. 1 = unigramas (palabras sueltas), 2 = bigramas (pares), "
         "3 = trigramas (de 3 en 3), 4 = 4-gramas.",
    "buscar": "BUSCAR: escribe una palabra para ver solo los n-gramas que la contienen. Ej. 'zorro'.",
    "tabla": "TABLA: cada renglon es un n-grama distinto y cuantas veces aparece. Esta ordenada del mas "
             "repetido al menos repetido. <s> = inicio de oracion, </s> = fin de oracion.",
    "conteo": "CONTEO: 'total' = cuantos n-gramas hay contando los repetidos.  'distintos' = cuantos "
              "n-gramas diferentes hay.  'Aparecen 1 sola vez' = los que nunca se repiten (con N mas "
              "grande casi todos son unicos).",
    "frase": "FRASE: escribe cualquier frase y se parte automaticamente en n-gramas. Sirve para ver "
             "como se forman los grupos de palabras.",
    "marcas": "MARCAS <s> y </s>: el libro agrega <s> al inicio y </s> al final de cada oracion para "
              "saber que palabras empiezan y terminan una oracion. Puedes quitarlas para ver solo las palabras.",
    "prob": "PROBABILIDAD DE UNA FRASE: se multiplica la probabilidad de cada bigrama.  "
            "P(palabra | anterior) = C(anterior palabra) / C(anterior), donde C = cuantas veces aparece "
            "en los cuentos. Si un bigrama nunca aparece, la frase completa vale 0.",
    "laplace": "LAPLACE (suavizado add-1): se le suma 1 a todos los conteos para que ningun bigrama valga 0. "
               "P = (C(anterior palabra) + 1) / (C(anterior) + V), V = tamano del vocabulario.",
    "predecir": "PREDECIR: escribe el inicio de una frase y el modelo dice que palabras suelen venir "
                "despues, segun los cuentos (como el autocompletar del celular).",
    "generar": "GENERAR: el modelo inventa una frase eligiendo cada palabra al azar segun que tan seguido "
               "aparece despues de las anteriores. Con trigramas suena mas parecido a los cuentos.",
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("N-gramas interactivo - Cuentos")
        root.geometry("1150x760")
        root.configure(bg="white")
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background="white", font=("Segoe UI", 10))
        st.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=(14, 6))
        st.configure("Treeview", rowheight=24, font=("Consolas", 10))
        st.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        st.configure("Big.TButton", font=("Segoe UI", 10, "bold"))

        nb = ttk.Notebook(root)
        nb.place(x=10, y=10, width=1130, height=640)
        self.tab_contar(nb)
        self.tab_partir(nb)
        self.tab_prob(nb)
        self.tab_predecir(nb)

        self.lbl_ayuda = tk.Label(root, text="", bg="#fff8dc", fg="#333", justify="left", anchor="nw",
                                  wraplength=1100, font=("Segoe UI", 10), relief="solid", bd=1, padx=10, pady=8)
        self.lbl_ayuda.place(x=10, y=660, width=1130, height=88)
        self.ayuda("inicio")

    # ---------- utilidades ----------
    def ayuda(self, clave):
        self.lbl_ayuda.config(text=AYUDA[clave])

    def hover(self, w, clave):
        w.bind("<Enter>", lambda e: self.ayuda(clave), add="+")
        return w

    def titulo(self, padre, texto, nota):
        ttk.Label(padre, text=texto, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(padre, text=nota, foreground="#666").pack(anchor="w", pady=(0, 8))

    def selector_n(self, padre, var, comando, clave="n"):
        f = ttk.Frame(padre)
        ttk.Label(f, text="N:", font=("Segoe UI", 10, "bold")).pack(side="left")
        for n in (1, 2, 3, 4):
            self.hover(ttk.Radiobutton(f, text=f"{n} ({NOMBRES_N[n]})", value=n, variable=var,
                                       command=comando), clave).pack(side="left", padx=4)
        self.hover(f, clave)
        return f

    # ---------- Pestana 1: contar ----------
    def tab_contar(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="1. Contar n-gramas")
        self.titulo(t, "Contar los n-gramas de los cuentos",
                    "Elige un cuento y N; la tabla muestra cada n-grama y cuantas veces aparece.")
        f = ttk.Frame(t); f.pack(fill="x")
        ttk.Label(f, text="Cuento:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.cuento = tk.StringVar(value=TODOS)
        cb = ttk.Combobox(f, textvariable=self.cuento, values=list(cuentos), state="readonly", width=30)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.contar())
        self.hover(cb, "cuento")
        self.tipo = tk.StringVar(value="palabras")
        self.n_contar = tk.IntVar(value=2)
        self.selector_n(f, self.n_contar, self.contar).pack(side="left", padx=12)
        for v, txt in (("palabras", "Palabras"), ("caracteres", "Caracteres")):
            self.hover(ttk.Radiobutton(f, text=txt, value=v, variable=self.tipo, command=self.contar),
                       "tipo").pack(side="left", padx=3)
        ttk.Label(f, text="Buscar:").pack(side="left")
        self.buscar = tk.StringVar()
        e = ttk.Entry(f, textvariable=self.buscar, width=16)
        e.pack(side="left", padx=4)
        self.buscar.trace_add("write", lambda *a: self.contar())
        self.hover(e, "buscar")

        cuerpo = ttk.Frame(t); cuerpo.pack(fill="both", expand=True, pady=10)
        izq = ttk.Frame(cuerpo); izq.pack(side="left", fill="both", expand=True)
        self.tabla = ttk.Treeview(izq, columns=("pos", "ng", "c", "bar"), show="headings", height=16)
        for col, txt, w, anc in [("pos", "#", 50, "e"), ("ng", "N-grama", 330, "w"),
                                 ("c", "Veces", 70, "e"), ("bar", "", 300, "w")]:
            self.tabla.heading(col, text=txt)
            self.tabla.column(col, width=w, anchor=anc)
        sb = ttk.Scrollbar(izq, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=sb.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.hover(self.tabla, "tabla")

        der = ttk.Frame(cuerpo, padding=(16, 0)); der.pack(side="left", fill="y")
        ttk.Label(der, text="Conteo", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.lbl_conteo = tk.Label(der, text="", bg="white", justify="left", font=("Consolas", 11), anchor="nw")
        self.lbl_conteo.pack(anchor="w")
        self.hover(self.lbl_conteo, "conteo")
        self.contar()

    def contar(self):
        n = self.n_contar.get()
        ors = cuentos[self.cuento.get()]
        car = self.tipo.get() == "caracteres"
        lista = ngramas_caracteres(ors, n) if car else ngramas(ors, n)
        c = Counter(lista)
        filtro = self.buscar.get().strip().lower()
        self.tabla.delete(*self.tabla.get_children())
        maximo = max(c.values()) if c else 1
        for i, (g, v) in enumerate(c.most_common(), start=1):
            texto = a_texto(g)
            if filtro and (filtro not in texto if car else filtro not in g):
                continue
            self.tabla.insert("", "end", values=(i, texto, v, "█" * max(1, round(30 * v / maximo))))
        palabras = sum(len(o) for o in ors)
        unicos = sum(1 for v in c.values() if v == 1)
        self.lbl_conteo.config(text=(
            f"Cuento: {self.cuento.get()}\n\n"
            f"Oraciones:        {len(ors)}\n"
            f"Palabras:         {palabras}\n\n"
            f"{NOMBRES_N[n]} de {'letras' if car else 'palabras'}:\n"
            f"  total:          {len(lista)}\n"
            f"  distintos:      {len(c)}\n"
            f"  aparecen 1 vez: {unicos}\n\n"
            f"Mas repetido:\n  '{a_texto(c.most_common(1)[0][0])}'\n  ({c.most_common(1)[0][1]} veces)"))

    # ---------- Pestana 2: partir una frase ----------
    def tab_partir(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="2. Partir una frase")
        self.titulo(t, "Partir una frase en n-gramas",
                    "Escribe cualquier frase y mira como se divide en grupos de palabras.")
        self.frase = tk.StringVar(value="El perro corre rápido")
        e = ttk.Entry(t, textvariable=self.frase, font=("Segoe UI", 12))
        e.pack(fill="x")
        self.frase.trace_add("write", lambda *a: self.partir())
        self.hover(e, "frase")
        self.marcas = tk.BooleanVar(value=False)
        fo = ttk.Frame(t); fo.pack(fill="x", pady=6)
        self.tipo_p = tk.StringVar(value="palabras")
        for v, txt in (("palabras", "Grupos de palabras"), ("caracteres", "Grupos de caracteres (letras)")):
            self.hover(ttk.Radiobutton(fo, text=txt, value=v, variable=self.tipo_p, command=self.partir),
                       "tipo").pack(side="left", padx=4)
        self.hover(ttk.Checkbutton(fo, text="Agregar marcas de inicio <s> y fin </s> (como en el libro)",
                                   variable=self.marcas, command=self.partir), "marcas").pack(side="left", padx=20)
        self.cv_partir = tk.Canvas(t, bg="white", highlightthickness=0)
        self.cv_partir.pack(fill="both", expand=True)
        self.hover(self.cv_partir, "n")
        self.partir()

    def partir(self):
        cv = self.cv_partir
        cv.delete("all")
        palabras = re.findall(r"[a-záéíóúüñ]+", self.frase.get().lower())
        colores = {1: "#dbeafe", 2: "#dcfce7", 3: "#fef3c7", 4: "#fce7f3"}
        y = 10
        for n in (1, 2, 3, 4):
            car = self.tipo_p.get() == "caracteres"
            if car:
                grupos = ngramas_caracteres([palabras], n)
            else:
                grupos = ngramas([palabras], n, marcas=self.marcas.get()) if palabras else []
            unidad = ("letra" if car else "palabra") + ("s" if n > 1 else "")
            linea = ". ".join(a_texto(g) for g in grupos)
            if len(linea) > 95:
                linea = linea[:95] + "..."
            cv.create_text(5, y, text=f"{NOMBRES_N[n]} ({len(grupos)}), cada grupo contiene {n} {unidad}:   {linea}" + ("." if grupos and not linea.endswith("...") else ""),
                           anchor="nw", font=("Segoe UI", 10, "bold"))
            y += 26
            x = 10
            for g in grupos:
                txt = a_texto(g)
                ancho = 9 * len(txt) + 18
                if x + ancho > 1090:
                    x, y = 10, y + 34
                cv.create_rectangle(x, y, x + ancho, y + 26, fill=colores[n], outline="#94a3b8")
                cv.create_text(x + ancho / 2, y + 13, text=txt, font=("Consolas", 10))
                x += ancho + 8
            y += 44

    # ---------- Pestana 3: probabilidad ----------
    def tab_prob(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="3. Probabilidad de una frase")
        self.titulo(t, "Probabilidad de una frase con bigramas",
                    "P(palabra | anterior) = C(anterior palabra) / C(anterior), contando en todos los cuentos.")
        self.frase_p = tk.StringVar(value="el robot cerró los ojos")
        e = ttk.Entry(t, textvariable=self.frase_p, font=("Segoe UI", 12))
        e.pack(fill="x")
        self.frase_p.trace_add("write", lambda *a: self.probabilidad())
        self.hover(e, "prob")
        f = ttk.Frame(t); f.pack(fill="x", pady=4)
        ttk.Label(f, text="Prueba:").pack(side="left")
        for ej in ("había una vez un zorro", "el zorro cerró los ojos", "la niña del faro", "el perro come pizza"):
            ttk.Button(f, text=ej, command=lambda ej=ej: self.frase_p.set(ej)).pack(side="left", padx=3)

        ors = cuentos[TODOS]
        self.uni = Counter()
        for o in ors:
            self.uni.update([INICIO] + o)
        self.big = Counter(ngramas(ors, 2))
        self.V = len({p for o in ors for p in o}) + 1

        self.tabla_p = ttk.Treeview(t, columns=("b", "cb", "ca", "p", "pl"), show="headings", height=11)
        for col, txt, w in [("b", "Bigrama", 220), ("cb", "C(anterior palabra)", 160), ("ca", "C(anterior)", 120),
                            ("p", "P sin suavizado", 220), ("pl", "P con Laplace", 220)]:
            self.tabla_p.heading(col, text=txt)
            self.tabla_p.column(col, width=w, anchor="center")
        self.tabla_p.pack(fill="x", pady=8)
        self.tabla_p.tag_configure("cero", background="#fee2e2")
        self.hover(self.tabla_p, "prob")
        self.lbl_p = tk.Label(t, text="", bg="white", justify="left", font=("Consolas", 12), anchor="w")
        self.lbl_p.pack(anchor="w")
        self.hover(self.lbl_p, "laplace")
        self.probabilidad()

    def probabilidad(self):
        palabras = re.findall(r"[a-záéíóúüñ]+", self.frase_p.get().lower())
        self.tabla_p.delete(*self.tabla_p.get_children())
        if not palabras:
            self.lbl_p.config(text="")
            return
        tokens = [INICIO] + palabras + [FIN]
        log_mle, log_lap, cero = 0.0, 0.0, False
        for prev, w in zip(tokens, tokens[1:]):
            cb, ca = self.big[(prev, w)], self.uni[prev]
            p = cb / ca if ca else 0.0
            pl = (cb + 1) / (ca + self.V)
            if p == 0:
                cero = True
            else:
                log_mle += math.log(p)
            log_lap += math.log(pl)
            self.tabla_p.insert("", "end", tags=("cero",) if p == 0 else (),
                                values=(f"{prev} {w}", cb, ca, f"{cb}/{ca} = {p:.4f}" if ca else "0 (nunca aparece)",
                                        f"{cb + 1}/{ca + self.V} = {pl:.4f}"))
        p_mle = 0.0 if cero else math.exp(log_mle)
        self.lbl_p.config(text=(
            f"Probabilidad de la frase (multiplicando los bigramas):\n"
            f"  sin suavizado: {p_mle:.3e}" + ("   <- vale 0 porque hay un bigrama que nunca aparece (en rojo)" if cero else "") +
            f"\n  con Laplace:   {math.exp(log_lap):.3e}\n"
            f"(notacion cientifica: 1.2e-05 = 0.000012)"))

    # ---------- Pestana 4: predecir y generar ----------
    def tab_predecir(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="4. Predecir y generar")
        self.titulo(t, "Predecir la siguiente palabra", "Escribe el inicio de una frase (como el autocompletar del celular).")
        self.inicio_p = tk.StringVar(value="había una")
        e = ttk.Entry(t, textvariable=self.inicio_p, font=("Segoe UI", 12))
        e.pack(fill="x")
        self.inicio_p.trace_add("write", lambda *a: self.predecir())
        self.hover(e, "predecir")
        self.lbl_pred = tk.Label(t, text="", bg="white", justify="left", font=("Consolas", 11), anchor="nw")
        self.lbl_pred.pack(anchor="w", pady=6)
        self.hover(self.lbl_pred, "predecir")

        ors = cuentos[TODOS]
        self.sig = {2: {}, 3: {}}
        for n in (2, 3):
            for g, c in Counter(ngramas(ors, n)).items():
                self.sig[n].setdefault(g[:-1], Counter())[g[-1]] += c

        ttk.Separator(t).pack(fill="x", pady=8)
        self.titulo(t, "Generar texto al azar", "El modelo inventa frases usando lo que aprendio de los cuentos.")
        f = ttk.Frame(t); f.pack(fill="x")
        self.n_gen = tk.IntVar(value=3)
        for n in (2, 3):
            self.hover(ttk.Radiobutton(f, text=f"con {NOMBRES_N[n].lower()}", value=n, variable=self.n_gen),
                       "generar").pack(side="left", padx=6)
        b = ttk.Button(f, text="Generar 3 frases", style="Big.TButton", command=self.generar)
        b.pack(side="left", padx=12)
        self.hover(b, "generar")
        self.txt_gen = tk.Text(t, height=7, wrap="word", font=("Segoe UI", 11), relief="solid", bd=1)
        self.txt_gen.pack(fill="x", pady=6)
        self.hover(self.txt_gen, "generar")
        self.predecir()
        self.generar()

    def predecir(self):
        palabras = re.findall(r"[a-záéíóúüñ]+", self.inicio_p.get().lower())
        hist = [INICIO, INICIO] + palabras
        lineas = []
        for n, desc in ((3, "usando las 2 ultimas palabras (trigramas)"), (2, "usando la ultima palabra (bigramas)")):
            ctx = tuple(hist[-(n - 1):])
            opciones = self.sig[n].get(ctx)
            lineas.append(f"Despues de '{' '.join(ctx)}' {desc}:")
            if not opciones:
                lineas.append("   (esa combinacion nunca aparece en los cuentos)")
            else:
                total = sum(opciones.values())
                for w, c in opciones.most_common(5):
                    nombre = "[fin de oracion]" if w == FIN else w
                    lineas.append(f"   {nombre:<18} {c}/{total} = {c / total:5.1%}  " + "█" * round(20 * c / total))
            lineas.append("")
        self.lbl_pred.config(text="\n".join(lineas))

    def generar(self):
        n = self.n_gen.get()
        frases = []
        for _ in range(3):
            hist, salida = [INICIO] * (n - 1), []
            while len(salida) < 30:
                opciones = self.sig[n][tuple(hist[-(n - 1):])]
                w = random.choices(list(opciones), weights=list(opciones.values()))[0]
                if w == FIN:
                    break
                salida.append(w)
                hist.append(w)
            frases.append("• " + " ".join(salida).capitalize() + ".")
        self.txt_gen.delete("1.0", "end")
        self.txt_gen.insert("1.0", "\n".join(frases))


if __name__ == "__main__":
    import sys
    root = tk.Tk()
    app = App(root)
    if "--prueba" in sys.argv:
        for n in (1, 2, 3, 4):
            app.n_contar.set(n); app.contar()
            print(n, app.lbl_conteo.cget("text").split("\n")[5:9])
        app.buscar.set("zorro"); print("filas con zorro:", len(app.tabla.get_children())); app.buscar.set("")
        app.tipo.set("caracteres"); app.n_contar.set(2); app.contar(); print("car2:", app.lbl_conteo.cget("text").split(chr(10))[5:12])
        app.buscar.set("rr"); print("filas rr:", [app.tabla.item(i)["values"][1] for i in app.tabla.get_children()]); app.buscar.set(""); app.tipo.set("palabras"); app.contar()
        app.tipo_p.set("caracteres"); app.partir(); app.tipo_p.set("palabras"); app.partir()
        app.frase_p.set("el zorro cerró los ojos"); print(app.lbl_p.cget("text"))
        app.inicio_p.set("había una"); print(app.lbl_pred.cget("text"))
        app.generar(); print(app.txt_gen.get("1.0", "end"))
        root.update(); root.after(200, root.destroy)
    root.mainloop()
