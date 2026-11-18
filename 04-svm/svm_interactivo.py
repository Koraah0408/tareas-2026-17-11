# SVM interactivo: clasificar libros por genero.
#   python svm_interactivo.py
#
# Izquierda: configuras el modelo (kernel, C, % de prueba, palabras) y ves que tan bien clasifica.
# Derecha: escribes (o eliges) la descripcion de un libro EN INGLES y el modelo predice su genero.
# Abajo: un cuadro amarillo te explica cualquier control con solo pasar el mouse encima.
import os
import re
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

AQUI = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(AQUI, "libros_generos.csv"))
GENEROS = sorted(df.genero.unique())
COLORES = {"Ciencia ficcion": "#3b82f6", "Fantasia": "#a855f7", "Historica": "#b45309",
           "Misterio": "#475569", "Romance": "#ec4899", "Terror": "#dc2626"}

EJEMPLOS = {
    "Fantasia": "A young girl discovers she is a witch and must use her magic to defeat an evil "
                "sorcerer who wants to rule the kingdom with his army of dragons.",
    "Misterio": "Detective Morgan investigates the murder of a wealthy banker found dead in his "
                "locked study. Every suspect has an alibi, but the clues point to someone close "
                "to the family.",
    "Ciencia ficcion": "The crew of a starship travels to a distant planet where they find an "
                       "alien civilization and advanced technology that could save humanity.",
    "Terror": "Something evil lives in the old house. At night the family hears screams, and one "
              "by one they are haunted by the ghost of a murdered child.",
    "Romance": "She never expected to fall in love with the charming stranger who moved into "
               "town, but their passion grows despite her family's disapproval of the marriage.",
    "Historica": "During the reign of King Henry VIII, a young servant at the royal court "
                 "witnesses the political intrigues of the Tudor dynasty and the war with France.",
}

AYUDA = {
    "inicio": "QUE ES ESTO?  Un SVM (Maquina de Vectores de Soporte) que aprende a clasificar libros "
              "en 6 generos leyendo su descripcion. Usa 1,654 libros reales (CMU Book Summary "
              "Dataset). Pasa el mouse sobre cualquier control para ver que hace.",
    "kernel": "KERNEL: la forma de la frontera que separa los generos.\n"
              "linear = fronteras rectas (lo mejor para texto, rapido).  rbf = fronteras curvas.  "
              "poly = polinomio (con texto suele salir peor).",
    "C": "C: que tanto castiga los errores al entrenar.\n"
         "Bajo (0.01) = modelo muy simple, casi no aprende.  Alto (100) = intenta acertar todos los "
         "libros de entrenamiento y puede memorizar en vez de aprender (sobreajuste).",
    "prueba": "% DE PRUEBA: que parte de los libros se aparta para examinar al modelo. Esos libros NO "
              "se usan para entrenar. Ej. 20% = entrena con 80% (1,323 libros) y se examina con "
              "20% (331 libros).",
    "palabras": "PALABRAS (vocabulario): cuantas palabras distintas toma en cuenta el modelo "
                "(las mas frecuentes). Cada libro se convierte en una lista de numeros, uno por "
                "palabra (TF-IDF). Pocas palabras = le falta informacion.",
    "entrenar": "ENTRENAR: vuelve a entrenar el modelo con la configuracion actual. Tambien se "
                "entrena solo cuando mueves un control.",
    "exactitud": "EXACTITUD: de los libros de prueba (que el modelo nunca vio), que porcentaje "
                 "clasifico bien. Al azar seria ~17% (1 de 6 generos).",
    "matriz": "MATRIZ DE CONFUSION: renglon = genero real, columna = lo que predijo el modelo. "
              "La diagonal (en verde) son los aciertos; lo demas son confusiones. Ej. Romance "
              "se confunde mucho con Historica.",
    "texto": "DESCRIPCION: escribe aqui la descripcion de un libro EN INGLES (el dataset esta en "
             "ingles) y presiona 'Predecir genero'.",
    "ejemplos": "EJEMPLOS: pone la descripcion de un libro inventado de ese genero para probar "
                "al modelo.",
    "predecir": "PREDECIR: el modelo lee la descripcion, la convierte a numeros y dice a que "
                "genero se parece mas.",
    "puntajes": "PUNTAJES: que tan fuerte 'vota' el modelo por cada genero (relativo, suma 100%). "
                "El mas alto es la prediccion.",
    "claves": "PALABRAS CLAVE: las palabras de tu descripcion que mas pesan, y con que genero "
              "las asocia el modelo segun los libros de entrenamiento.",
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("SVM interactivo - Clasificar libros por genero")
        root.geometry("1250x800")
        root.configure(bg="white")
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background="white", font=("Segoe UI", 10))
        st.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        st.configure("Big.TButton", font=("Segoe UI", 11, "bold"))

        self.modelo = self.vect = None
        self.pendiente = None

        izq = ttk.LabelFrame(root, text=" 1. Configura y entrena el modelo ", padding=10)
        izq.place(x=10, y=10, width=520, height=690)
        der = ttk.LabelFrame(root, text=" 2. Prueba con un libro ", padding=10)
        der.place(x=540, y=10, width=700, height=690)

        # ---------- Controles ----------
        self.kernel = tk.StringVar(value="linear")
        f = ttk.Frame(izq); f.pack(fill="x")
        ttk.Label(f, text="Kernel:", font=("Segoe UI", 10, "bold")).pack(side="left")
        for k in ("linear", "rbf", "poly"):
            rb = ttk.Radiobutton(f, text=k, value=k, variable=self.kernel, command=self.programar)
            rb.pack(side="left", padx=6)
            self.ayuda_en(rb, "kernel")
        self.ayuda_en(f, "kernel")
        self.nota(izq, "forma de la frontera entre generos")

        self.c_exp = tk.DoubleVar(value=0)
        self.lbl_c = self.slider(izq, "C:", self.c_exp, -2, 2, "C",
                                 "cuanto castiga los errores (bajo = simple, alto = memoriza)")
        self.prueba = tk.IntVar(value=20)
        self.lbl_p = self.slider(izq, "% de prueba:", self.prueba, 10, 50, "prueba",
                                 "libros apartados para examinar al modelo")
        self.palabras = tk.IntVar(value=5000)
        self.lbl_w = self.slider(izq, "Palabras:", self.palabras, 100, 10000, "palabras",
                                 "tamano del vocabulario que usa el modelo")

        b = ttk.Button(izq, text="Entrenar", style="Big.TButton", command=self.entrenar)
        b.pack(fill="x", pady=(8, 4))
        self.ayuda_en(b, "entrenar")

        self.lbl_estado = ttk.Label(izq, text="", foreground="#555", wraplength=490)
        self.lbl_estado.pack(anchor="w")
        self.lbl_acc = tk.Label(izq, text="", font=("Segoe UI", 20, "bold"), bg="white", fg="#15803d")
        self.lbl_acc.pack(anchor="w", pady=(4, 0))
        self.ayuda_en(self.lbl_acc, "exactitud")
        self.lbl_acc2 = ttk.Label(izq, text="", foreground="#555")
        self.lbl_acc2.pack(anchor="w")

        ttk.Label(izq, text="Matriz de confusion (renglon = real, columna = predicho):",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.grid_cm = tk.Frame(izq, bg="white")
        self.grid_cm.pack(anchor="w")
        self.ayuda_en(self.grid_cm, "matriz")

        # ---------- Prueba con un libro ----------
        ttk.Label(der, text="Descripcion del libro (en ingles):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.txt = tk.Text(der, height=6, wrap="word", font=("Segoe UI", 10), relief="solid", bd=1)
        self.txt.pack(fill="x")
        self.ayuda_en(self.txt, "texto")

        fe = ttk.Frame(der); fe.pack(fill="x", pady=6)
        ttk.Label(fe, text="Ejemplos:").pack(side="left")
        for g in GENEROS:
            eb = tk.Button(fe, text=g, bg=COLORES[g], fg="white", relief="flat", padx=6,
                           font=("Segoe UI", 9, "bold"), command=lambda g=g: self.poner_ejemplo(g))
            eb.pack(side="left", padx=2)
            self.ayuda_en(eb, "ejemplos")

        bp = ttk.Button(der, text="Predecir genero", style="Big.TButton", command=self.predecir)
        bp.pack(fill="x", pady=4)
        self.ayuda_en(bp, "predecir")

        self.lbl_pred = tk.Label(der, text="", font=("Segoe UI", 22, "bold"), bg="white")
        self.lbl_pred.pack(anchor="w", pady=(6, 2))
        ttk.Label(der, text="Puntaje para cada genero:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.cv = tk.Canvas(der, height=170, bg="white", highlightthickness=0)
        self.cv.pack(fill="x")
        self.ayuda_en(self.cv, "puntajes")
        ttk.Label(der, text="Palabras clave detectadas (palabra -> genero con el que se asocia):",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))
        self.lbl_claves = ttk.Label(der, text="", wraplength=670, justify="left")
        self.lbl_claves.pack(anchor="w")
        self.ayuda_en(self.lbl_claves, "claves")

        # ---------- Cuadro de ayuda ----------
        self.lbl_ayuda = tk.Label(root, text="", bg="#fff8dc", fg="#333", justify="left",
                                  anchor="nw", wraplength=1200, font=("Segoe UI", 10),
                                  relief="solid", bd=1, padx=10, pady=8)
        self.lbl_ayuda.place(x=10, y=708, width=1230, height=82)
        self.mostrar_ayuda("inicio")

        self.entrenar()
        self.poner_ejemplo("Fantasia")
        self.predecir()

    # ---------- utilidades de interfaz ----------
    def nota(self, padre, texto):
        ttk.Label(padre, text=texto, foreground="#777", font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 6))

    def slider(self, padre, nombre, var, a, b, clave, nota):
        f = ttk.Frame(padre); f.pack(fill="x")
        ttk.Label(f, text=nombre, width=12, font=("Segoe UI", 10, "bold")).pack(side="left")
        s = ttk.Scale(f, from_=a, to=b, variable=var, command=lambda _: self.cambio(), length=290)
        s.pack(side="left")
        lbl = ttk.Label(f, text="", width=8)
        lbl.pack(side="left", padx=6)
        for w in (f, s, lbl):
            self.ayuda_en(w, clave)
        self.nota(padre, nota)
        return lbl

    def ayuda_en(self, widget, clave):
        widget.bind("<Enter>", lambda e: self.mostrar_ayuda(clave), add="+")

    def mostrar_ayuda(self, clave):
        self.lbl_ayuda.config(text=AYUDA[clave])

    def valores(self):
        C = 10 ** self.c_exp.get()
        p = int(round(self.prueba.get()))
        w = int(round(self.palabras.get() / 100) * 100)
        self.lbl_c.config(text=f"{C:.2f}")
        self.lbl_p.config(text=f"{p}%")
        self.lbl_w.config(text=f"{w}")
        return C, p, w

    def cambio(self):
        self.valores()
        self.programar()

    def programar(self):
        # espera a que dejes de mover el control para entrenar (evita entrenar 50 veces)
        if self.pendiente:
            self.root.after_cancel(self.pendiente)
        self.pendiente = self.root.after(500, self.entrenar)

    # ---------- modelo ----------
    def entrenar(self):
        self.pendiente = None
        C, p, w = self.valores()
        kernel = self.kernel.get()
        self.lbl_estado.config(text="Entrenando...")
        self.root.update_idletasks()

        X_tr, X_te, y_tr, y_te = train_test_split(df.descripcion, df.genero, test_size=p / 100,
                                                  random_state=42, stratify=df.genero)
        self.vect = TfidfVectorizer(stop_words="english", max_features=w, sublinear_tf=True)
        Xtr = self.vect.fit_transform(X_tr)
        self.modelo = SVC(kernel=kernel, C=C, gamma="scale", degree=3)
        self.modelo.fit(Xtr, y_tr)
        pred = self.modelo.predict(self.vect.transform(X_te))
        acc = accuracy_score(y_te, pred)

        # genero con el que mas se asocia cada palabra (promedio TF-IDF por genero)
        y_arr = np.array(y_tr)
        medias = np.vstack([np.asarray(Xtr[y_arr == g].mean(axis=0)).ravel() for g in GENEROS])
        self.genero_palabra = medias.argmax(axis=0)

        self.lbl_estado.config(text=f"Entrenado con {len(X_tr)} libros  |  examinado con {len(X_te)}\n"
                                    f"kernel={kernel}  C={C:.2f}  |  vectores de soporte: {self.modelo.support_.size}")
        self.lbl_acc.config(text=f"Exactitud: {acc * 100:.1f}%")
        self.lbl_acc2.config(text=f"Acerto {int(round(acc * len(y_te)))} de {len(y_te)} libros de prueba "
                                  f"(al azar seria ~17%)")
        self.dibujar_matriz(confusion_matrix(y_te, pred, labels=GENEROS))
        if self.txt.get("1.0", "end").strip():
            self.predecir()

    def dibujar_matriz(self, cm):
        for w in self.grid_cm.winfo_children():
            w.destroy()
        cortos = [g[:9] for g in GENEROS]
        tk.Label(self.grid_cm, text="", bg="white", width=13).grid(row=0, column=0)
        for j, g in enumerate(cortos):
            tk.Label(self.grid_cm, text=g, bg="white", font=("Segoe UI", 8, "bold"), width=7).grid(row=0, column=j + 1)
        maximo = cm.max() or 1
        for i, g in enumerate(GENEROS):
            tk.Label(self.grid_cm, text=g, bg="white", anchor="e", font=("Segoe UI", 9, "bold"),
                     width=13).grid(row=i + 1, column=0, sticky="e")
            for j in range(len(GENEROS)):
                v = cm[i, j]
                t = v / maximo
                if i == j:
                    color = f"#{int(255 - 150 * t):02x}{int(255 - 60 * t):02x}{int(255 - 150 * t):02x}"
                else:
                    color = f"#{255:02x}{int(255 - 120 * t):02x}{int(255 - 120 * t):02x}"
                c = tk.Label(self.grid_cm, text=str(v), bg=color, width=7, height=1,
                             font=("Segoe UI", 10, "bold" if i == j else "normal"), relief="solid", bd=1)
                c.grid(row=i + 1, column=j + 1)
                self.ayuda_en(c, "matriz")

    def poner_ejemplo(self, g):
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", EJEMPLOS[g])
        self.predecir()

    def predecir(self):
        texto = self.txt.get("1.0", "end").strip()
        if not texto or self.modelo is None:
            return
        x = self.vect.transform([texto])
        genero = self.modelo.predict(x)[0]
        self.lbl_pred.config(text=f"Genero predicho: {genero}", fg=COLORES[genero])

        puntos = self.modelo.decision_function(x)[0]          # un puntaje por genero
        e = np.exp(puntos - puntos.max())
        pct = e / e.sum() * 100
        self.cv.delete("all")
        ancho = 470
        for i, g in enumerate(self.modelo.classes_):
            y = 8 + i * 26
            self.cv.create_text(120, y + 9, text=g, anchor="e", font=("Segoe UI", 10, "bold"))
            self.cv.create_rectangle(130, y, 130 + ancho, y + 18, fill="#f1f5f9", outline="")
            self.cv.create_rectangle(130, y, 130 + ancho * pct[i] / 100, y + 18, fill=COLORES[g], outline="")
            self.cv.create_text(140 + ancho, y + 9, text=f"{pct[i]:.1f}%", anchor="w", font=("Segoe UI", 10))

        vocab = self.vect.vocabulary_
        pesos = x.toarray()[0]
        palabras = {p for p in re.findall(r"[a-z]+", texto.lower()) if p in vocab}
        top = sorted(palabras, key=lambda p: -pesos[vocab[p]])[:10]
        if top:
            self.lbl_claves.config(text="   ".join(f"{p} -> {GENEROS[self.genero_palabra[vocab[p]]]}" for p in top))
        else:
            self.lbl_claves.config(text="(ninguna palabra de la descripcion esta en el vocabulario; escribe en ingles)")


if __name__ == "__main__":
    import sys
    root = tk.Tk()
    app = App(root)
    if "--prueba" in sys.argv:       # solo para comprobar sin quedarse abierto
        for k in ("rbf", "poly", "linear"):
            app.kernel.set(k); app.entrenar()
            print(k, app.lbl_acc.cget("text"), "|", app.lbl_pred.cget("text"))
        app.c_exp.set(-2); app.entrenar(); print("C=0.01", app.lbl_acc.cget("text"))
        app.c_exp.set(0); app.palabras.set(100); app.entrenar(); print("100 palabras", app.lbl_acc.cget("text"))
        app.palabras.set(5000); app.prueba.set(40); app.entrenar(); print("40% prueba", app.lbl_acc.cget("text"), app.lbl_acc2.cget("text"))
        app.prueba.set(20); app.entrenar()
        for g in GENEROS:
            app.poner_ejemplo(g); print("ejemplo", g, "->", app.lbl_pred.cget("text"))
        print("claves:", app.lbl_claves.cget("text"))
        root.update()
        root.after(300, root.destroy)
    root.mainloop()
