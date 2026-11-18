# Red neuronal interactiva
#   python red_neuronal_interactiva.py
#
# Red neuronal con TensorFlow/Keras, con controles:
#   eliges el conjunto de entrenamiento, el tipo de red (perceptron de 1 neurona o multicapa),
#   la tasa de aprendizaje y las epocas; ves la grafica de perdida mientras entrena, los pesos
#   que aprendio y puedes pedirle una prediccion.
# Pasa el mouse sobre cualquier cosa y el cuadro amarillo de abajo te explica que es.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # menos mensajes de TensorFlow
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Conjuntos de entrenamiento: nombre -> (entradas, salidas, unidad entrada, unidad salida, formula real)
DATASETS = {
    "Kilometros a millas": (
        [0, 1, 5, 10, 21, 42, 80, 160], None, "km", "millas", lambda x: x * 0.621371, "millas = km x 0.621371"),
    "Celsius a Fahrenheit": (
        [-40, -10, 0, 8, 15, 22, 38], [-40, 14, 32, 46, 59, 72, 100], "°C", "°F",
        lambda x: x * 1.8 + 32, "°F = °C x 1.8 + 32"),
    "Kilogramos a libras": (
        [0, 1, 2, 5, 10, 20, 50, 100], None, "kg", "lb", lambda x: x * 2.20462, "lb = kg x 2.20462"),
    "Metros a pies": (
        [0, 1, 3, 5, 10, 25, 50, 100], None, "m", "pies", lambda x: x * 3.28084, "pies = m x 3.28084"),
}

AYUDA = {
    "inicio": "QUE ES ESTO?  Una red neuronal aprende una relacion entre numeros SIN que le digas la formula: "
              "solo le das ejemplos (entrada -> salida) y ajusta sus pesos hasta equivocarse lo menos posible. "
              "Pasa el mouse sobre cualquier cosa para ver que es.",
    "dataset": "CONJUNTO DE ENTRENAMIENTO: los ejemplos que se le dan a la red (por ejemplo kilometros -> millas). "
               "La red nunca ve la formula, solo los pares de numeros.",
    "tabla": "DATOS DE ENTRENAMIENTO: cada renglon es un ejemplo (entrada y la salida correcta). Con esto aprende.",
    "modelo": "TIPO DE RED:\n- Perceptron (1 neurona): salida = w x entrada + b "
              "(capa = Dense(units=1)).\n- Multicapa (3 + 3 + 1): 2 capas ocultas de 3 neuronas y 1 de salida. "
              "Tambien se llama perceptron multicapa (MLP).",
    "lr": "TASA DE APRENDIZAJE: que tan grandes son los ajustes en cada paso (por defecto: Adam(0.1)). "
          "Muy chica = aprende lento. Muy grande = brinca y puede no aprender.",
    "epocas": "EPOCAS: cuantas veces la red repasa todos los ejemplos (por defecto: 1000). "
              "Mas epocas = mas practica, pero tarda mas.",
    "entrenar": "ENTRENAR: crea la red y la entrena con la configuracion elegida. La grafica se va dibujando mientras aprende.",
    "perdida": "GRAFICA DE PERDIDA: que tanto se equivoca la red en cada epoca (error cuadratico medio). "
               "Empieza alta porque los pesos son al azar y baja conforme aprende. Esta en escala logaritmica.",
    "pesos": "PESOS: lo que aprendio la red. En el perceptron son solo 2 numeros: w (peso) y b (sesgo), y "
             "deberian parecerse a la formula real. En la multicapa son muchos numeros y no se leen tan facil.",
    "predecir": "PREDECIR: escribe un numero que la red NO vio al entrenar y compara su respuesta con el valor real.",
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Red neuronal interactiva")
        root.geometry("1200x800")
        root.configure(bg="white")
        st = ttk.Style()
        st.theme_use("clam")
        st.configure(".", background="white", font=("Segoe UI", 10))
        st.configure("Treeview", rowheight=22)
        st.configure("Big.TButton", font=("Segoe UI", 11, "bold"))
        self.tab_red(root)
        self.lbl_ayuda = tk.Label(root, text="", bg="#fff8dc", fg="#333", justify="left", anchor="nw",
                                  wraplength=1150, font=("Segoe UI", 10), relief="solid", bd=1, padx=10, pady=8)
        self.lbl_ayuda.place(x=10, y=700, width=1180, height=90)
        self.ayuda("inicio")

    def ayuda(self, clave):
        self.lbl_ayuda.config(text=AYUDA[clave])

    def hover(self, w, clave):
        w.bind("<Enter>", lambda e: self.ayuda(clave), add="+")
        return w

    # =================== Red neuronal ===================
    def tab_red(self, root):
        t = ttk.Frame(root, padding=10)
        t.place(x=10, y=10, width=1180, height=680)
        ttk.Label(t, text="Red neuronal", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        t = ttk.Frame(t)
        t.pack(fill="both", expand=True, pady=(6, 0))
        izq = ttk.Frame(t, width=360); izq.pack(side="left", fill="y"); izq.pack_propagate(False)
        der = ttk.Frame(t); der.pack(side="left", fill="both", expand=True, padx=(12, 0))

        ttk.Label(izq, text="Conjunto de entrenamiento", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.ds = tk.StringVar(value=list(DATASETS)[0])
        cb = ttk.Combobox(izq, textvariable=self.ds, values=list(DATASETS), state="readonly", width=40)
        cb.pack(anchor="w", pady=(2, 4))
        cb.bind("<<ComboboxSelected>>", lambda e: self.mostrar_datos())
        self.hover(cb, "dataset")
        self.tabla = ttk.Treeview(izq, columns=("x", "y"), show="headings", height=8)
        self.tabla.pack(fill="x")
        self.hover(self.tabla, "tabla")

        ttk.Label(izq, text="Tipo de red", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.modelo_tipo = tk.StringVar(value="perceptron")
        for v, txt in (("perceptron", "Perceptron: 1 neurona (w x entrada + b)"),
                       ("multicapa", "Multicapa: 3 + 3 + 1 neuronas")):
            self.hover(ttk.Radiobutton(izq, text=txt, value=v, variable=self.modelo_tipo), "modelo").pack(anchor="w")

        self.lr_exp = tk.DoubleVar(value=-1)          # tasa = 10^x
        self.lbl_lr = self.slider(izq, "Tasa de aprendizaje", self.lr_exp, -3, 0, "lr",
                                  lambda v: f"{10 ** float(v):.3f}")
        self.epocas = tk.IntVar(value=1000)
        self.lbl_ep = self.slider(izq, "Epocas", self.epocas, 50, 2000, "epocas", lambda v: f"{int(float(v))}")

        b = ttk.Button(izq, text="Entrenar", style="Big.TButton", command=self.entrenar)
        b.pack(fill="x", pady=8)
        self.hover(b, "entrenar")
        self.lbl_estado = ttk.Label(izq, text="", foreground="#555", wraplength=340)
        self.lbl_estado.pack(anchor="w")

        # grafica
        self.fig = Figure(figsize=(7.4, 3.6), dpi=90)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=der)
        self.canvas.get_tk_widget().pack(fill="x")
        self.hover(self.canvas.get_tk_widget(), "perdida")

        abajo = ttk.Frame(der); abajo.pack(fill="both", expand=True, pady=6)
        fp = ttk.Frame(abajo); fp.pack(side="left", fill="both", expand=True)
        ttk.Label(fp, text="Lo que aprendio la red (pesos)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lbl_pesos = tk.Label(fp, text="(entrena la red)", bg="white", justify="left", anchor="nw",
                                  font=("Consolas", 10))
        self.lbl_pesos.pack(anchor="w", fill="both")
        self.hover(self.lbl_pesos, "pesos")

        fpr = ttk.Frame(abajo, width=330); fpr.pack(side="left", fill="y"); fpr.pack_propagate(False)
        ttk.Label(fpr, text="Hagamos una prediccion", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        fe = ttk.Frame(fpr); fe.pack(anchor="w", pady=4)
        self.valor = tk.StringVar(value="100")
        e = ttk.Entry(fe, textvariable=self.valor, width=10, font=("Segoe UI", 12))
        e.pack(side="left")
        self.lbl_unidad = ttk.Label(fe, text="")
        self.lbl_unidad.pack(side="left", padx=4)
        bp = ttk.Button(fe, text="Predecir", command=self.predecir)
        bp.pack(side="left", padx=6)
        e.bind("<Return>", lambda ev: self.predecir())
        for w in (e, bp):
            self.hover(w, "predecir")
        self.lbl_pred = tk.Label(fpr, text="", bg="white", justify="left", anchor="nw", font=("Segoe UI", 11))
        self.lbl_pred.pack(anchor="w")
        self.hover(self.lbl_pred, "predecir")

        self.modelo = None
        self.mostrar_datos()
        self.dibujar_perdida([])

    def slider(self, padre, nombre, var, a, b, clave, fmt):
        ttk.Label(padre, text=nombre, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        f = ttk.Frame(padre); f.pack(fill="x")
        lbl = ttk.Label(f, text=fmt(var.get()), width=8)
        s = ttk.Scale(f, from_=a, to=b, variable=var, length=260, command=lambda v: lbl.config(text=fmt(v)))
        s.pack(side="left")
        lbl.pack(side="left", padx=6)
        for w in (f, s, lbl):
            self.hover(w, clave)
        return lbl

    def datos(self):
        xs, ys, ue, us, f, formula = DATASETS[self.ds.get()]
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float) if ys is not None else np.round(f(x), 3)
        return x, y, ue, us, f, formula

    def mostrar_datos(self):
        x, y, ue, us, _, formula = self.datos()
        self.tabla.heading("x", text=f"Entrada ({ue})")
        self.tabla.heading("y", text=f"Salida ({us})")
        self.tabla.delete(*self.tabla.get_children())
        for a, b in zip(x, y):
            self.tabla.insert("", "end", values=(f"{a:g}", f"{b:g}"))
        self.lbl_unidad.config(text=ue)
        self.lbl_estado.config(text=f"Formula real (la red NO la conoce): {formula}")
        self.modelo = None
        self.lbl_pesos.config(text="(entrena la red)")
        self.lbl_pred.config(text="")

    def dibujar_perdida(self, hist):
        self.ax.clear()
        self.ax.set_title("Perdida (error) en cada epoca")
        self.ax.set_xlabel("# Epoca")
        self.ax.set_ylabel("Magnitud de perdida")
        if hist:
            self.ax.plot(range(1, len(hist) + 1), hist, color="tab:blue")
            self.ax.set_yscale("log")
        self.ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def entrenar(self):
        import tensorflow as tf
        x, y, ue, us, f, formula = self.datos()
        lr = 10 ** self.lr_exp.get()
        epocas = int(self.epocas.get())
        tipo = self.modelo_tipo.get()
        self.lbl_estado.config(text="Entrenando..."); self.root.update()

        if tipo == "perceptron":
            self.capas = [tf.keras.layers.Dense(units=1)]
        else:
            self.capas = [tf.keras.layers.Dense(units=3), tf.keras.layers.Dense(units=3), tf.keras.layers.Dense(units=1)]
        self.modelo = tf.keras.Sequential([tf.keras.Input(shape=(1,))] + self.capas)
        self.modelo.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mean_squared_error")

        hist = []
        app = self

        class EnVivo(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                hist.append(logs["loss"])
                if epoch % max(1, epocas // 25) == 0:
                    app.dibujar_perdida(hist)
                    app.lbl_estado.config(text=f"Entrenando... epoca {epoch + 1}/{epocas}  perdida = {logs['loss']:.4f}")
                    app.root.update()

        self.modelo.fit(x, y, epochs=epocas, verbose=0, callbacks=[EnVivo()])
        self.dibujar_perdida(hist)
        self.lbl_estado.config(text=f"Modelo entrenado! ({epocas} epocas)  perdida final = {hist[-1]:.5f}\n"
                                    f"Formula real (la red NO la conoce): {formula}")

        if tipo == "perceptron":
            w, b = self.capas[0].get_weights()
            self.lbl_pesos.config(text=(
                f"Perceptron: salida = w x entrada + b\n\n"
                f"  w (peso)  = {w[0][0]:.5f}\n"
                f"  b (sesgo) = {b[0]:.5f}\n\n"
                f"La red aprendio:  {us} = {ue} x {w[0][0]:.4f} + {b[0]:.4f}\n"
                f"Formula real:     {formula}"))
        else:
            lineas = ["Multicapa: muchos pesos (no se leen tan facil)"]
            for i, capa in enumerate(self.capas, 1):
                w, b = capa.get_weights()
                nombre = "salida" if i == len(self.capas) else f"oculta{i}"
                lineas.append(f"\n{nombre}: w = {np.round(w.astype(float), 3).tolist()}")
                lineas.append(f"{' ' * len(nombre)}  b = {np.round(b.astype(float), 3).tolist()}")
            self.lbl_pesos.config(text="\n".join(lineas))
        self.predecir()

    def predecir(self):
        if self.modelo is None:
            self.lbl_pred.config(text="Primero entrena la red.")
            return
        try:
            v = float(self.valor.get())
        except ValueError:
            self.lbl_pred.config(text="Escribe un numero.")
            return
        _, _, ue, us, f, _ = self.datos()
        p = float(self.modelo.predict(np.array([[v]]), verbose=0)[0][0])
        real = f(v)
        err = abs(p - real)
        self.lbl_pred.config(text=f"{v:g} {ue}  ->\n  la red dice:  {p:.3f} {us}\n  valor real:   {real:.3f} {us}\n"
                                  f"  diferencia:   {err:.3f}", fg="#15803d" if err < max(1, abs(real) * 0.02) else "#b91c1c")


if __name__ == "__main__":
    import sys
    root = tk.Tk()
    app = App(root)
    if "--prueba" in sys.argv:
        for tipo in ("perceptron", "multicapa"):
            app.modelo_tipo.set(tipo); app.epocas.set(500); app.entrenar()
            print(tipo, "|", app.lbl_estado.cget("text").split("\n")[0], "|", app.lbl_pred.cget("text").replace("\n", " "))
        print(app.lbl_pesos.cget("text")[:200])
        app.ds.set("Celsius a Fahrenheit"); app.mostrar_datos(); app.modelo_tipo.set("perceptron"); app.epocas.set(1000); app.entrenar()
        print("celsius:", app.lbl_pesos.cget("text").replace("\n", " | "))
        root.update(); root.after(200, root.destroy)
    root.mainloop()
