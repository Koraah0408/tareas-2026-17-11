# NLTK interactivo (presentacion myNLTK)
#   python nltk_interactivo.py
#
# Pestanas:
#   1. Buscar en libros ..... concordance, similar y conteo de una palabra en Moby Dick, Jane Austen, etc.
#   2. Frecuencias .......... palabras mas repetidas de un libro, con grafica (FreqDist).
#   3. WordNet .............. significados, sinonimos, "tipos de", "es un tipo de", partes y opuestos.
#   4. Etiquetar ............ partes de la oracion (sustantivo, verbo...), chunking y entidades (personas, lugares).
#   5. Clasificar genero .... un clasificador Naive Bayes adivina si un nombre es de hombre o de mujer.
#   6. Chatbot .............. un tutor de NLTK para platicar.
# Pasa el mouse sobre cualquier cosa y el cuadro amarillo de abajo te explica que es.
# (La primera vez tarda unos segundos en abrir porque carga los libros.)
import io
import random
import string
import contextlib
import tkinter as tk
from tkinter import ttk

import nltk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from nltk.corpus import wordnet as wn, names, stopwords, gutenberg, genesis, inaugural, nps_chat, webtext, treebank
from nltk.chat.util import Chat, reflections

# Los mismos 9 textos de "from nltk.book import *", pero cada uno se carga SOLO cuando lo eliges
# (cargar los 9 al inicio tarda mucho).
FUENTES = {
    "text1": ("Moby Dick by Herman Melville 1851", lambda: gutenberg.words("melville-moby_dick.txt")),
    "text2": ("Sense and Sensibility by Jane Austen 1811", lambda: gutenberg.words("austen-sense.txt")),
    "text3": ("The Book of Genesis", lambda: genesis.words("english-kjv.txt")),
    "text4": ("Inaugural Address Corpus", lambda: inaugural.words()),
    "text5": ("Chat Corpus", lambda: nps_chat.words()),
    "text6": ("Monty Python and the Holy Grail", lambda: webtext.words("grail.txt")),
    "text7": ("Wall Street Journal", lambda: treebank.words()),
    "text8": ("Personals Corpus", lambda: webtext.words("singles.txt")),
    "text9": ("The Man Who Was Thursday by G . K . Chesterton 1908", lambda: gutenberg.words("chesterton-thursday.txt")),
}
NOMBRES_LIBROS = {k: f"{k}: {nombre}" for k, (nombre, _) in FUENTES.items()}
_cache_libros = {}


def cargar_libro(clave):
    if clave not in _cache_libros:
        nombre, palabras = FUENTES[clave]
        print(f"Cargando {nombre}...", flush=True)
        _cache_libros[clave] = nltk.Text(palabras(), name=nombre)
    return _cache_libros[clave]

ETIQUETAS = {  # significado en espanol de las etiquetas de nltk.pos_tag
    "NN": "sustantivo", "NNS": "sustantivo plural", "NNP": "nombre propio", "NNPS": "nombre propio plural",
    "VB": "verbo (base)", "VBD": "verbo pasado", "VBG": "verbo -ing (gerundio)", "VBN": "participio",
    "VBP": "verbo presente", "VBZ": "verbo presente 3a persona", "JJ": "adjetivo", "JJR": "adjetivo comparativo",
    "JJS": "adjetivo superlativo", "RB": "adverbio", "RBR": "adverbio comparativo", "DT": "determinante (the, a)",
    "IN": "preposicion", "CC": "conjuncion (and, or)", "PRP": "pronombre", "PRP$": "pronombre posesivo",
    "TO": "la palabra 'to'", "MD": "verbo modal (can, will)", "CD": "numero", "WP": "pronombre wh (who)",
    "WDT": "determinante wh (which)", "WRB": "adverbio wh (when, where)", "EX": "existencial (there)",
    "UH": "interjeccion", "RP": "particula", "POS": "posesivo ('s)", ".": "punto", ",": "coma",
}
ENTIDADES = {"PERSON": "Persona", "GPE": "Lugar (pais/ciudad)", "ORGANIZATION": "Organizacion",
             "LOCATION": "Lugar", "FACILITY": "Instalacion", "GSP": "Lugar"}

AYUDA = {
    "inicio": "QUE ES NLTK?  Natural Language Toolkit: una libreria de Python para que la computadora trabaje con "
              "lenguaje humano. Cada pestana es una parte de la presentacion myNLTK. Pasa el mouse sobre cualquier "
              "cosa para ver que es.",
    "libro": "LIBRO: NLTK trae 9 textos para practicar (text1 = Moby Dick, text2 = Sense and Sensibility, "
             "text4 = discursos de presidentes de EE.UU., etc.). Estan en ingles.",
    "palabra": "PALABRA: escribe una palabra en ingles (por ejemplo monstrous, whale, love, freedom).",
    "concordance": "CONCORDANCE: muestra cada vez que aparece la palabra en el libro, con lo que tiene a los lados "
                   "(su contexto). Sirve para ver como se usa.",
    "similar": "SIMILAR: palabras que aparecen en contextos parecidos a la tuya. En Moby Dick 'monstrous' se parece "
               "a 'curious', y en Jane Austen a 'very', porque ella la usa como 'muy'.",
    "contar": "CONTAR: cuantas veces aparece la palabra en el libro y que porcentaje del texto es.",
    "frec": "FRECUENCIAS (FreqDist): cuenta cuantas veces aparece cada palabra. Las mas comunes suelen ser "
            "palabras vacias como 'the', 'of', 'and'.",
    "vacias": "QUITAR PALABRAS VACIAS: ignora palabras que casi no dicen nada (the, of, and, a...) y los signos de "
              "puntuacion, para que se vean las palabras importantes del libro.",
    "topn": "CUANTAS: cuantas palabras mostrar en la grafica.",
    "longitud": "LONGITUD: en vez de contar palabras, cuenta cuantas palabras hay de 1 letra, de 2, de 3, etc.",
    "colloc": "COLLOCATIONS: pares de palabras que aparecen juntas mas de lo normal (ej. 'United States').",
    "wn": "WORDNET: un diccionario para computadoras. Cada significado de una palabra es un 'synset' "
          "(ej. car.n.01 = car, sustantivo, significado 1).",
    "synsets": "SIGNIFICADOS: una palabra puede tener varios. 'car' puede ser automovil, vagon de tren, cabina de "
               "elevador... Haz clic en uno para ver su informacion.",
    "detalle": "DETALLE: sinonimos (lemmas), definicion, ejemplos, hiperonimos ('es un tipo de'), hiponimos "
               "('tipos de'), partes (meronimos), de que forma parte (holonimos) y opuestos (antonimos).",
    "sim": "SIMILITUD: que tan parecidas son dos palabras segun el arbol de WordNet (0 = nada, 1 = iguales). "
           "Ej. dog-cat se parecen mas que dog-car.",
    "frase": "FRASE: escribe una oracion en ingles.",
    "pos": "ETIQUETAR (pos_tag): dice que es cada palabra: sustantivo, verbo, adjetivo... Las etiquetas vienen "
           "del Penn Treebank (NN = sustantivo, VB = verbo, JJ = adjetivo...).",
    "chunk": "CHUNKING: agrupa palabras en frases nominales (NP) con la regla <DT>?<JJ>*<NN.*>+ "
             "= determinante opcional + adjetivos + sustantivo. Ej. 'the little yellow dog'.",
    "ner": "ENTIDADES (ne_chunk): encuentra nombres de personas, lugares y organizaciones.",
    "genero": "CLASIFICAR GENERO: un clasificador Naive Bayes entrenado con 7,944 nombres en ingles. Mira la ultima "
              "letra y las ultimas 2 letras del nombre. Ej. nombres que terminan en 'a' suelen ser de mujer.",
    "nombre": "NOMBRE: escribe un nombre (de preferencia en ingles) y el modelo adivina si es de hombre o de mujer.",
    "rasgos": "RASGOS MAS INFORMATIVOS: lo que mas le ayuda al modelo a decidir. 'female : male = 38 : 1' significa "
              "que ese rasgo aparece 38 veces mas en nombres de mujer.",
    "chat": "CHATBOT: un sistema de dialogo hecho con nltk.chat. Responde segun patrones (expresiones regulares). "
            "Escribe 'hola' y preguntale por tokenizar, etiquetar, wordnet, chunking o clasificar.",
}

PARES_CHAT = [
    (r"(hola|buenas|hi|hello)(.*)", ["Hola! Soy tu tutor de NLTK. Preguntame por: tokenizar, etiquetar, wordnet, chunking o clasificar."]),
    (r"(.*)tokeni(.*)", ["Para tokenizar usa nltk.word_tokenize('texto') y para oraciones nltk.sent_tokenize('texto')."]),
    (r"(.*)etiquet(.*)|(.*)pos(.*)", ["Para etiquetar partes de la oracion: nltk.pos_tag(nltk.word_tokenize('texto'))."]),
    (r"(.*)wordnet(.*)|(.*)sinonim(.*)", ["Con WordNet: from nltk.corpus import wordnet as wn; wn.synsets('car')."]),
    (r"(.*)chunk(.*)", ["Un chunker se hace con nltk.RegexpParser('NP: {<DT>?<JJ>*<NN>}')."]),
    (r"(.*)clasific(.*)", ["Para clasificar: nltk.NaiveBayesClassifier.train(train_set) y luego classifier.classify(rasgos)."]),
    (r"(.*)frecuencia(.*)|(.*)freqdist(.*)", ["FreqDist cuenta palabras: nltk.FreqDist(texto).most_common(10)."]),
    (r"me llamo (.*)", ["Mucho gusto, %1. Que quieres aprender de NLTK?"]),
    (r"(.*)gracias(.*)", ["De nada! Algo mas?"]),
    (r"(adios|bye|salir)(.*)", ["Hasta luego!"]),
    (r"(.*)", ["No entendi. Prueba con: tokenizar, etiquetar, wordnet, chunking, clasificar o frecuencias.",
               "Puedes reformularlo? Se de tokenizar, etiquetar, wordnet, chunking, clasificar y frecuencias."]),
]


def capturar(funcion, *args, **kwargs):
    """Ejecuta una funcion de NLTK que imprime en pantalla y regresa lo que imprimio."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        funcion(*args, **kwargs)
    return buf.getvalue().strip()


class App:
    def __init__(self, root):
        self.root = root
        root.title("NLTK interactivo")
        root.geometry("1200x800")
        root.configure(bg="white")
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background="white", font=("Segoe UI", 10))
        st.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        st.configure("Treeview", rowheight=22)
        st.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        nb = ttk.Notebook(root)
        nb.place(x=10, y=10, width=1180, height=680)
        self.tab_buscar(nb)
        self.tab_frecuencias(nb)
        self.tab_wordnet(nb)
        self.tab_etiquetar(nb)
        self.tab_genero(nb)
        self.tab_chat(nb)
        self.lbl_ayuda = tk.Label(root, text="", bg="#fff8dc", fg="#333", justify="left", anchor="nw",
                                  wraplength=1150, font=("Segoe UI", 10), relief="solid", bd=1, padx=10, pady=8)
        self.lbl_ayuda.place(x=10, y=700, width=1180, height=90)
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

    def caja_texto(self, padre, alto, clave):
        f = ttk.Frame(padre)
        t = tk.Text(f, height=alto, wrap="none", font=("Consolas", 10), relief="solid", bd=1)
        sy = ttk.Scrollbar(f, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sy.set)
        t.pack(side="left", fill="both", expand=True)
        sy.pack(side="left", fill="y")
        self.hover(t, clave)
        return f, t

    def escribir(self, caja, texto):
        caja.config(state="normal")
        caja.delete("1.0", "end")
        caja.insert("1.0", texto)
        caja.config(state="disabled")

    def selector_libro(self, padre, var, comando=None):
        cb = ttk.Combobox(padre, textvariable=var, values=list(NOMBRES_LIBROS.values()), state="readonly", width=48)
        if comando:
            cb.bind("<<ComboboxSelected>>", lambda e: comando())
        self.hover(cb, "libro")
        return cb

    def libro(self, var):
        return cargar_libro(var.get().split(":")[0])

    # ---------- 1. Buscar en libros ----------
    def tab_buscar(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="1. Buscar en libros")
        self.titulo(t, "Buscar una palabra en los libros de NLTK",
                    "concordance (contexto), similar (palabras parecidas) y cuantas veces aparece.")
        f = ttk.Frame(t); f.pack(fill="x")
        self.libro_b = tk.StringVar(value=NOMBRES_LIBROS["text1"])
        self.selector_libro(f, self.libro_b).pack(side="left")
        ttk.Label(f, text="   Palabra:").pack(side="left")
        self.palabra_b = tk.StringVar(value="monstrous")
        e = ttk.Entry(f, textvariable=self.palabra_b, width=18, font=("Segoe UI", 11))
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.buscar())
        self.hover(e, "palabra")
        b = ttk.Button(f, text="Buscar", command=self.buscar)
        b.pack(side="left", padx=6)
        self.hover(b, "concordance")

        ttk.Label(t, text="Conteo", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.lbl_conteo_b = ttk.Label(t, text="")
        self.lbl_conteo_b.pack(anchor="w")
        self.hover(self.lbl_conteo_b, "contar")
        ttk.Label(t, text="Concordance (la palabra con su contexto)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        fc, self.txt_conc = self.caja_texto(t, 14, "concordance")
        fc.pack(fill="both", expand=True)
        ttk.Label(t, text="Similar (palabras que se usan en contextos parecidos)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        fs, self.txt_sim = self.caja_texto(t, 3, "similar")
        self.txt_sim.config(wrap="word")
        fs.pack(fill="x")
        self.lbl_conteo_b.config(text="Cargando Moby Dick...")
        self.root.after(100, self.buscar)   # primero se muestra la ventana y luego se busca

    def buscar(self):
        texto = self.libro(self.libro_b)
        w = self.palabra_b.get().strip()
        if not w:
            return
        n = sum(1 for x in texto if x.lower() == w.lower())
        self.lbl_conteo_b.config(text=f"'{w}' aparece {n} veces en {texto.name}  "
                                      f"({n / len(texto) * 100:.4f}% de sus {len(texto):,} palabras)")
        lineas = texto.concordance_list(w, width=110, lines=60)
        conc = "\n".join(l.line for l in lineas) if lineas else "(no aparece en este libro)"
        self.escribir(self.txt_conc, f"{len(lineas)} coincidencias mostradas\n\n{conc}")
        sim = capturar(texto.similar, w)
        self.escribir(self.txt_sim, sim if sim else "(no se encontraron palabras parecidas)")

    # ---------- 2. Frecuencias ----------
    def tab_frecuencias(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="2. Frecuencias")
        self.titulo(t, "Frecuencia de palabras (FreqDist)", "Las palabras que mas se repiten en un libro.")
        f = ttk.Frame(t); f.pack(fill="x")
        self.libro_f = tk.StringVar(value=NOMBRES_LIBROS["text1"])
        self.selector_libro(f, self.libro_f, self.frecuencias).pack(side="left")
        self.vacias = tk.BooleanVar(value=True)
        self.hover(ttk.Checkbutton(f, text="Quitar palabras vacias y signos", variable=self.vacias,
                                   command=self.frecuencias), "vacias").pack(side="left", padx=12)
        self.modo_f = tk.StringVar(value="palabras")
        for v, txt in (("palabras", "Palabras"), ("longitud", "Longitud de palabra")):
            self.hover(ttk.Radiobutton(f, text=txt, value=v, variable=self.modo_f, command=self.frecuencias),
                       "longitud" if v == "longitud" else "frec").pack(side="left", padx=4)
        ttk.Label(f, text="   Cuantas:").pack(side="left")
        self.topn = tk.IntVar(value=20)
        sp = ttk.Spinbox(f, from_=5, to=50, textvariable=self.topn, width=5, command=self.frecuencias)
        sp.pack(side="left")
        self.hover(sp, "topn")

        cuerpo = ttk.Frame(t); cuerpo.pack(fill="both", expand=True, pady=8)
        self.fig_f = Figure(figsize=(8.2, 4.4), dpi=90)
        self.ax_f = self.fig_f.add_subplot(111)
        self.cv_f = FigureCanvasTkAgg(self.fig_f, master=cuerpo)
        self.cv_f.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.hover(self.cv_f.get_tk_widget(), "frec")
        der = ttk.Frame(cuerpo, width=300, padding=(10, 0)); der.pack(side="left", fill="y"); der.pack_propagate(False)
        self.lbl_f = tk.Label(der, text="", bg="white", justify="left", anchor="nw", font=("Consolas", 10), wraplength=280)
        self.lbl_f.pack(anchor="w")
        self.hover(self.lbl_f, "frec")
        ttk.Label(der, text="Collocations", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.lbl_col = tk.Label(der, text="", bg="white", justify="left", anchor="nw", wraplength=280)
        self.lbl_col.pack(anchor="w")
        self.hover(self.lbl_col, "colloc")
        self.vacias_set = set(stopwords.words("english")) | set(string.punctuation) | {"--", "''", "``", "'s", "n't"}
        # la grafica se calcula la primera vez que abres esta pestana (asi la ventana abre mas rapido)
        self.frec_lista = False
        nb.bind("<<NotebookTabChanged>>", lambda e: self.al_cambiar_pestana(nb), add="+")

    def al_cambiar_pestana(self, nb):
        if not self.frec_lista and nb.index(nb.select()) == 1:
            self.frec_lista = True
            self.frecuencias()

    def frecuencias(self):
        texto = self.libro(self.libro_f)
        try:
            n = max(5, min(50, int(self.topn.get())))
        except (ValueError, tk.TclError):
            n = 20
        palabras = [w.lower() for w in texto]
        if self.vacias.get():
            palabras = [w for w in palabras if w not in self.vacias_set and w.isalpha()]
        ax = self.ax_f
        ax.clear()
        if self.modo_f.get() == "longitud":
            fd = nltk.FreqDist(len(w) for w in palabras)
            items = sorted(fd.items())[:n]
            ax.bar([str(k) for k, _ in items], [v for _, v in items], color="#3b82f6")
            ax.set_xlabel("Letras por palabra")
            ax.set_title(f"Cuantas palabras hay de cada longitud - {texto.name}", fontsize=10)
            info = f"Longitud mas comun: {fd.max()} letras\n({fd[fd.max()]:,} palabras, {fd.freq(fd.max()):.1%})"
        else:
            fd = nltk.FreqDist(palabras)
            items = fd.most_common(n)
            ax.bar([w for w, _ in items], [v for _, v in items], color="#3b82f6")
            ax.tick_params(axis="x", rotation=60)
            ax.set_title(f"Las {n} palabras mas repetidas - {texto.name}", fontsize=10)
            info = f"Mas repetida: '{fd.max()}'\n({fd[fd.max()]:,} veces, {fd.freq(fd.max()):.2%})"
        ax.set_ylabel("Veces")
        ax.grid(axis="y", alpha=0.3)
        self.fig_f.tight_layout()
        self.cv_f.draw_idle()
        self.lbl_f.config(text=(f"Libro: {texto.name}\n\nPalabras totales: {len(texto):,}\n"
                                f"Palabras distintas: {len(set(texto)):,}\nContadas: {len(palabras):,}\n\n{info}"))
        if not hasattr(texto, "_col_cache"):
            texto._col_cache = capturar(texto.collocations, num=12)
        self.lbl_col.config(text=texto._col_cache.replace("\n", " "))

    # ---------- 3. WordNet ----------
    def tab_wordnet(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="3. WordNet")
        self.titulo(t, "WordNet: el diccionario de NLTK",
                    "Escribe una palabra en ingles y elige uno de sus significados.")
        f = ttk.Frame(t); f.pack(fill="x")
        ttk.Label(f, text="Palabra:").pack(side="left")
        self.palabra_w = tk.StringVar(value="car")
        e = ttk.Entry(f, textvariable=self.palabra_w, width=18, font=("Segoe UI", 11))
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.wordnet())
        self.hover(e, "wn")
        ttk.Button(f, text="Buscar", command=self.wordnet).pack(side="left", padx=4)
        for ej in ("car", "dog", "tree", "walk", "good", "bank"):
            ttk.Button(f, text=ej, width=6, command=lambda ej=ej: (self.palabra_w.set(ej), self.wordnet())).pack(side="left", padx=2)

        cuerpo = ttk.Frame(t); cuerpo.pack(fill="both", expand=True, pady=8)
        izq = ttk.Frame(cuerpo, width=430); izq.pack(side="left", fill="y"); izq.pack_propagate(False)
        ttk.Label(izq, text="Significados (synsets)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lista_syn = ttk.Treeview(izq, columns=("s", "d"), show="headings", height=14)
        self.lista_syn.heading("s", text="Synset"); self.lista_syn.heading("d", text="Definicion")
        self.lista_syn.column("s", width=130); self.lista_syn.column("d", width=290)
        self.lista_syn.pack(fill="both", expand=True)
        self.lista_syn.bind("<<TreeviewSelect>>", lambda e: self.detalle_synset())
        self.hover(self.lista_syn, "synsets")

        fsim = ttk.LabelFrame(izq, text=" Similitud entre dos palabras ", padding=6)
        fsim.pack(fill="x", pady=(8, 0))
        self.sim1, self.sim2 = tk.StringVar(value="dog"), tk.StringVar(value="cat")
        ff = ttk.Frame(fsim); ff.pack(fill="x")
        for v in (self.sim1, self.sim2):
            en = ttk.Entry(ff, textvariable=v, width=12)
            en.pack(side="left", padx=3)
            en.bind("<Return>", lambda ev: self.similitud())
            self.hover(en, "sim")
        ttk.Button(ff, text="Comparar", command=self.similitud).pack(side="left", padx=4)
        self.lbl_sim = ttk.Label(fsim, text="", wraplength=400)
        self.lbl_sim.pack(anchor="w")
        self.hover(fsim, "sim")

        fd_, self.txt_wn = self.caja_texto(cuerpo, 20, "detalle")
        self.txt_wn.config(wrap="word")
        fd_.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.root.after(300, lambda: (self.wordnet(), self.similitud()))   # despues de mostrar la ventana

    def wordnet(self):
        w = self.palabra_w.get().strip().lower().replace(" ", "_")
        self.lista_syn.delete(*self.lista_syn.get_children())
        syns = wn.synsets(w)
        for s in syns:
            self.lista_syn.insert("", "end", iid=s.name(), values=(s.name(), s.definition()))
        if syns:
            self.lista_syn.selection_set(syns[0].name())
            self.detalle_synset()
        else:
            self.escribir(self.txt_wn, f"'{w}' no esta en WordNet (recuerda escribir en ingles).")

    def detalle_synset(self):
        sel = self.lista_syn.selection()
        if not sel:
            return
        s = wn.synset(sel[0])
        nombres = lambda lst: ", ".join(x.name() for x in lst) or "(ninguno)"
        antonimos = [a.name() for l in s.lemmas() for a in l.antonyms()]
        hipo = sorted(l.name() for h in s.hyponyms() for l in h.lemmas())
        camino = s.hypernym_paths()[0] if s.hypernym_paths() else []
        self.escribir(self.txt_wn, (
            f"SYNSET: {s.name()}\n"
            f"Definicion: {s.definition()}\n"
            f"Ejemplos: {'; '.join(s.examples()) or '(ninguno)'}\n\n"
            f"SINONIMOS (lemmas): {', '.join(s.lemma_names())}\n"
            f"OPUESTOS (antonimos): {', '.join(antonimos) or '(ninguno)'}\n\n"
            f"ES UN TIPO DE (hiperonimos): {nombres(s.hypernyms())}\n"
            f"Camino hasta lo mas general:\n   " + " -> ".join(x.name().split('.')[0] for x in reversed(camino)) + "\n\n"
            f"TIPOS DE {s.lemma_names()[0].upper()} (hiponimos, {len(hipo)}):\n   {', '.join(hipo[:40])}{' ...' if len(hipo) > 40 else ''}\n\n"
            f"PARTES (meronimos): {nombres(s.part_meronyms())}\n"
            f"HECHO DE (meronimos de sustancia): {nombres(s.substance_meronyms())}\n"
            f"FORMA PARTE DE (holonimos): {nombres(s.member_holonyms() + s.part_holonyms())}\n"
            f"IMPLICA (entailments, solo verbos): {nombres(s.entailments())}"))

    def similitud(self):
        a, b = self.sim1.get().strip().lower(), self.sim2.get().strip().lower()
        sa, sb = wn.synsets(a), wn.synsets(b)
        if not sa or not sb:
            self.lbl_sim.config(text="Alguna de las dos palabras no esta en WordNet.")
            return
        mejor = max(((x.path_similarity(y) or 0, x, y) for x in sa[:5] for y in sb[:5]), key=lambda t: t[0])
        comun = mejor[1].lowest_common_hypernyms(mejor[2])
        self.lbl_sim.config(text=(f"Similitud {a} - {b}: {mejor[0]:.3f}  (0 = nada, 1 = iguales)\n"
                                  f"Lo que tienen en comun: {comun[0].name() if comun else '?'}"))

    # ---------- 4. Etiquetar ----------
    def tab_etiquetar(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="4. Etiquetar frase")
        self.titulo(t, "Etiquetar una frase (en ingles)",
                    "Partes de la oracion (pos_tag), frases nominales (chunking) y entidades (ne_chunk).")
        self.frase_e = tk.StringVar(value="George Washington led the little army from Virginia to New York")
        e = ttk.Entry(t, textvariable=self.frase_e, font=("Segoe UI", 12))
        e.pack(fill="x")
        e.bind("<Return>", lambda ev: self.etiquetar())
        self.hover(e, "frase")
        f = ttk.Frame(t); f.pack(fill="x", pady=4)
        ttk.Button(f, text="Etiquetar", command=self.etiquetar).pack(side="left")
        for ej in ("The little yellow dog barked at the cat",
                   "Barack Obama visited Google in California",
                   "And now for something completely different"):
            ttk.Button(f, text=ej, command=lambda ej=ej: (self.frase_e.set(ej), self.etiquetar())).pack(side="left", padx=3)

        cuerpo = ttk.Frame(t); cuerpo.pack(fill="both", expand=True, pady=6)
        izq = ttk.Frame(cuerpo); izq.pack(side="left", fill="both", expand=True)
        ttk.Label(izq, text="Partes de la oracion (pos_tag)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tabla_pos = ttk.Treeview(izq, columns=("w", "t", "s"), show="headings", height=16)
        for c, txt, w in (("w", "Palabra", 150), ("t", "Etiqueta", 80), ("s", "Que es", 230)):
            self.tabla_pos.heading(c, text=txt); self.tabla_pos.column(c, width=w)
        self.tabla_pos.pack(fill="both", expand=True)
        self.hover(self.tabla_pos, "pos")
        der = ttk.Frame(cuerpo, padding=(12, 0)); der.pack(side="left", fill="both", expand=True)
        ttk.Label(der, text="Chunking: frases nominales (NP)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_chunk = tk.Label(der, text="", bg="white", justify="left", anchor="nw", font=("Consolas", 10), wraplength=520)
        self.lbl_chunk.pack(anchor="w", fill="x")
        self.hover(self.lbl_chunk, "chunk")
        ttk.Label(der, text="Entidades encontradas (ne_chunk)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(14, 0))
        self.lbl_ner = tk.Label(der, text="", bg="white", justify="left", anchor="nw", font=("Consolas", 11))
        self.lbl_ner.pack(anchor="w")
        self.hover(self.lbl_ner, "ner")
        self.chunker = nltk.RegexpParser("NP: {<DT>?<JJ.*>*<NN.*>+}")
        self.root.after(500, self.etiquetar)   # despues de mostrar la ventana

    def etiquetar(self):
        tokens = nltk.word_tokenize(self.frase_e.get())
        if not tokens:
            return
        tags = nltk.pos_tag(tokens)
        self.tabla_pos.delete(*self.tabla_pos.get_children())
        for w, tg in tags:
            self.tabla_pos.insert("", "end", values=(w, tg, ETIQUETAS.get(tg, "otro")))
        arbol = self.chunker.parse(tags)
        nps = [" ".join(w for w, _ in st.leaves()) for st in arbol.subtrees(lambda x: x.label() == "NP")]
        self.lbl_chunk.config(text="Regla: NP = <DT>?<JJ>*<NN>+\n\n" + ("\n".join(f"[NP] {n}" for n in nps) or "(ninguna)"))
        ents = [(st.label(), " ".join(w for w, _ in st.leaves()))
                for st in nltk.ne_chunk(tags).subtrees() if st.label() != "S"]
        self.lbl_ner.config(text="\n".join(f"{ENTIDADES.get(l, l):<22} {txt}" for l, txt in ents) or "(ninguna)")

    # ---------- 5. Clasificar genero ----------
    def tab_genero(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="5. Clasificar genero")
        self.titulo(t, "Clasificador de genero por nombre (Naive Bayes)",
                    "Aprende de 7,944 nombres en ingles mirando sus ultimas letras.")
        etiquetados = ([(n, "male") for n in names.words("male.txt")] +
                       [(n, "female") for n in names.words("female.txt")])
        random.seed(0)
        random.shuffle(etiquetados)
        rasgos = lambda n: {"ultima letra": n[-1:].lower(), "ultimas 2 letras": n[-2:].lower()}
        self.rasgos = rasgos
        datos = [(rasgos(n), g) for n, g in etiquetados]
        entrena, prueba = datos[500:], datos[:500]
        self.clf = nltk.NaiveBayesClassifier.train(entrena)
        acc = nltk.classify.accuracy(self.clf, prueba)

        f = ttk.Frame(t); f.pack(fill="x")
        ttk.Label(f, text="Nombre:").pack(side="left")
        self.nombre = tk.StringVar(value="Trinity")
        e = ttk.Entry(f, textvariable=self.nombre, width=20, font=("Segoe UI", 12))
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.clasificar())
        self.nombre.trace_add("write", lambda *a: self.clasificar())
        self.hover(e, "nombre")
        for ej in ("Neo", "Trinity", "Maria", "Carlos", "Sofia", "Kevin"):
            ttk.Button(f, text=ej, width=8, command=lambda ej=ej: self.nombre.set(ej)).pack(side="left", padx=2)
        self.lbl_gen = tk.Label(t, text="", bg="white", font=("Segoe UI", 22, "bold"))
        self.lbl_gen.pack(anchor="w", pady=10)
        self.hover(self.lbl_gen, "genero")
        self.lbl_gen2 = tk.Label(t, text="", bg="white", justify="left", font=("Consolas", 11), anchor="w")
        self.lbl_gen2.pack(anchor="w")
        self.hover(self.lbl_gen2, "genero")
        ttk.Label(t, text=f"Exactitud con 500 nombres que no vio al entrenar: {acc:.1%}",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        ttk.Label(t, text="Rasgos mas informativos:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        info = capturar(self.clf.show_most_informative_features, 8)
        lbl = tk.Label(t, text=info, bg="white", justify="left", anchor="w", font=("Consolas", 10))
        lbl.pack(anchor="w")
        self.hover(lbl, "rasgos")
        self.clasificar()

    def clasificar(self):
        n = self.nombre.get().strip()
        if not n:
            self.lbl_gen.config(text="")
            self.lbl_gen2.config(text="")
            return
        r = self.rasgos(n)
        dist = self.clf.prob_classify(r)
        g = dist.max()
        self.lbl_gen.config(text=f"{n} -> {'Hombre' if g == 'male' else 'Mujer'}",
                            fg="#2563eb" if g == "male" else "#db2777")
        self.lbl_gen2.config(text=(f"Rasgos que mira: ultima letra = '{r['ultima letra']}', "
                                   f"ultimas 2 letras = '{r['ultimas 2 letras']}'\n"
                                   f"Probabilidad: hombre {dist.prob('male'):.1%}  |  mujer {dist.prob('female'):.1%}"))

    # ---------- 6. Chatbot ----------
    def tab_chat(self, nb):
        t = ttk.Frame(nb, padding=12)
        nb.add(t, text="6. Chatbot")
        self.titulo(t, "Chatbot tutor de NLTK (sistema de dialogo)", "Hecho con nltk.chat: responde segun patrones.")
        self.bot = Chat(PARES_CHAT, reflections)
        fc, self.txt_chat = self.caja_texto(t, 20, "chat")
        self.txt_chat.config(wrap="word", font=("Segoe UI", 11))
        self.txt_chat.tag_configure("tu", foreground="#2563eb", font=("Segoe UI", 11, "bold"))
        self.txt_chat.tag_configure("bot", foreground="#15803d")
        fc.pack(fill="both", expand=True)
        f = ttk.Frame(t); f.pack(fill="x", pady=6)
        self.msg = tk.StringVar()
        e = ttk.Entry(f, textvariable=self.msg, font=("Segoe UI", 12))
        e.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda ev: self.enviar())
        self.hover(e, "chat")
        ttk.Button(f, text="Enviar", command=self.enviar).pack(side="left", padx=6)
        self.txt_chat.config(state="normal")
        self.txt_chat.insert("end", "Tutor: Hola! Preguntame por tokenizar, etiquetar, wordnet, chunking, clasificar o frecuencias.\n\n", "bot")
        self.txt_chat.config(state="disabled")

    def enviar(self):
        m = self.msg.get().strip()
        if not m:
            return
        resp = self.bot.respond(m.lower()) or "..."
        self.txt_chat.config(state="normal")
        self.txt_chat.insert("end", f"Tu: {m}\n", "tu")
        self.txt_chat.insert("end", f"Tutor: {resp}\n\n", "bot")
        self.txt_chat.see("end")
        self.txt_chat.config(state="disabled")
        self.msg.set("")


if __name__ == "__main__":
    import sys
    print("Abriendo NLTK interactivo, espera unos segundos...", flush=True)
    root = tk.Tk()
    app = App(root)
    print("Ventana lista.", flush=True)
    if "--prueba" in sys.argv:
        app.wordnet(); app.similitud(); app.etiquetar()
        app.frec_lista = True; app.frecuencias()
        print("buscar:", app.lbl_conteo_b.cget("text"))
        print("similar:", app.txt_sim.get("1.0", "end").strip()[:120])
        app.palabra_b.set("whale"); app.buscar(); print("whale:", app.lbl_conteo_b.cget("text"))
        print("frec:", app.lbl_f.cget("text").replace("\n", " | "))
        app.modo_f.set("longitud"); app.frecuencias(); print("long:", app.lbl_f.cget("text").split("\n\n")[-1].replace("\n", " "))
        app.libro_f.set(NOMBRES_LIBROS["text4"]); app.modo_f.set("palabras"); app.frecuencias(); print("colloc4:", app.lbl_col.cget("text")[:100])
        print("wn:", app.txt_wn.get("1.0", "3.0").strip())
        print("sim:", app.lbl_sim.cget("text"))
        print("pos:", [app.tabla_pos.item(i)["values"] for i in app.tabla_pos.get_children()][:4])
        print("chunk:", app.lbl_chunk.cget("text").replace("\n", " | "))
        print("ner:", app.lbl_ner.cget("text").replace("\n", " | "))
        for n in ("Neo", "Trinity", "Maria", "Carlos"):
            app.nombre.set(n); print("genero:", app.lbl_gen.cget("text"), app.lbl_gen2.cget("text").split("\n")[1])
        for m in ("hola", "como tokenizo?", "que es wordnet"):
            app.msg.set(m); app.enviar()
        print("chat:", app.txt_chat.get("1.0", "end").strip().replace("\n\n", " || ")[-250:])
        root.update(); root.after(200, root.destroy)
    root.mainloop()
