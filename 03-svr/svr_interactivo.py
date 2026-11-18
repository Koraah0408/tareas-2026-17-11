# SVR interactivo: mueve los controles y el modelo se vuelve a entrenar al instante.
#
#   python svr_interactivo.py
#
# Pasa el mouse sobre cualquier control y el cuadro amarillo te explica que hace.
# Controles:
#   - Kernel ............ forma de la curva que usa el SVR (rbf, linear, poly)
#   - C ................. que tanto castiga los errores (alto = se ajusta mas a los datos)
#   - Epsilon ........... margen de error que tolera sin castigar
#   - Ventana (meses) ... cuantos meses anteriores mira para predecir el siguiente
#   - Fin entrenamiento . ultimo ano que se usa para entrenar; lo demas se predice
#   - Modo .............. "Mes a mes": usa el precio real del mes anterior para cada prediccion
#                         "A ciegas": predice todo el periodo de golpe, sin ver los datos reales
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.widgets import Slider, RadioButtons, Button
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

AQUI = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(AQUI, "sp500_2010_2018.csv"), parse_dates=["Fecha"])
precios = df.SP500.values
retornos = np.diff(precios) / precios[:-1]   # retornos[i] = cambio del mes i al mes i+1

INICIAL = dict(kernel="rbf", C=1.0, epsilon=0.1, ventana=12, fin=2016, modo="Mes a mes")


def entrenar_y_predecir(kernel, C, epsilon, ventana, fin, modo):
    """Entrena con los meses hasta diciembre de 'fin' y predice el resto."""
    X, y, idx = [], [], []
    for i in range(ventana, len(retornos)):
        X.append(retornos[i - ventana:i])
        y.append(retornos[i])
        idx.append(i + 1)                      # indice del mes que se predice
    X, y, idx = np.array(X), np.array(y), np.array(idx)
    entrena = df.Fecha.values[idx] <= np.datetime64(f"{fin}-12-01")

    ex = StandardScaler().fit(X[entrena])
    ey = StandardScaler().fit(y[entrena].reshape(-1, 1))
    modelo = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma="scale", degree=3)
    modelo.fit(ex.transform(X[entrena]), ey.transform(y[entrena].reshape(-1, 1)).ravel())

    def predice_ret(ventana_ret):
        z = modelo.predict(ex.transform(np.array(ventana_ret).reshape(1, -1)))
        return ey.inverse_transform(z.reshape(-1, 1))[0, 0]

    meses = idx[~entrena]
    pred = []
    if modo == "Mes a mes":
        for m in meses:
            r = predice_ret(retornos[m - 1 - ventana:m - 1])
            pred.append(precios[m - 1] * (1 + r))
    else:  # A ciegas: cada prediccion se usa como si fuera el dato real del siguiente paso
        historia = list(retornos[:meses[0] - 1])
        precio = precios[meses[0] - 1]
        for m in meses:
            r = predice_ret(historia[-ventana:])
            historia.append(r)
            precio = precio * (1 + r)
            pred.append(precio)
    pred = np.array(pred)
    reales = precios[meses]
    mape = np.mean(np.abs(reales - pred) / reales) * 100
    mae = np.mean(np.abs(reales - pred))
    ss = np.sum((reales - reales.mean()) ** 2)
    r2 = 1 - np.sum((reales - pred) ** 2) / ss if ss > 0 else float("nan")
    return df.Fecha.values[meses], reales, pred, mape, mae, r2, len(modelo.support_), entrena.sum()


# ---------------- Ventana y graficas ----------------
fig = plt.figure(figsize=(14, 8.5))
fig.canvas.manager.set_window_title("SVR interactivo - S&P 500")
ax = fig.add_axes([0.06, 0.40, 0.65, 0.54])
ax_err = fig.add_axes([0.06, 0.07, 0.65, 0.23])

linea_hist, = ax.plot(df.Fecha, precios, color="gray", lw=1.5, label="Datos reales")
linea_real, = ax.plot([], [], color="tab:blue", marker="o", ms=4, label="Real (periodo a predecir)")
linea_pred, = ax.plot([], [], color="tab:red", marker="x", ls="--", label="Prediccion SVR")
corte = ax.axvline(df.Fecha[0], color="black", ls=":", label="Fin del entrenamiento")
zona = None
ax.set_ylabel("Precio del S&P 500")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
barras = None
ax_err.xaxis.set_major_locator(mdates.YearLocator())
ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_err.set_ylabel("Error %")
ax_err.grid(alpha=0.3)
texto = fig.text(0.745, 0.97, "", va="top", fontsize=10, family="monospace")

# ---------------- Controles (cada uno con su explicacion) ----------------
GRIS = "#555555"
t_kernel = fig.text(0.745, 0.765, "Kernel", fontsize=10, weight="bold")
radio_kernel = RadioButtons(fig.add_axes([0.745, 0.665, 0.11, 0.095]), ("rbf", "linear", "poly"))
t_modo = fig.text(0.865, 0.765, "Modo", fontsize=10, weight="bold")
radio_modo = RadioButtons(fig.add_axes([0.865, 0.665, 0.125, 0.095]), ("Mes a mes", "A ciegas"))

s_C = Slider(fig.add_axes([0.81, 0.61, 0.13, 0.025]), "C", -2, 2, valinit=0)          # C = 10^x
n_C = fig.text(0.745, 0.585, "cuanto castiga los errores al entrenar", fontsize=8, color=GRIS)
s_eps = Slider(fig.add_axes([0.81, 0.535, 0.13, 0.025]), "Epsilon", 0.01, 1.0, valinit=INICIAL["epsilon"])
n_eps = fig.text(0.745, 0.51, "margen de error que se tolera", fontsize=8, color=GRIS)
s_vent = Slider(fig.add_axes([0.81, 0.46, 0.13, 0.025]), "Ventana", 1, 24, valinit=INICIAL["ventana"], valstep=1)
n_vent = fig.text(0.745, 0.435, "meses anteriores que mira para predecir", fontsize=8, color=GRIS)
s_fin = Slider(fig.add_axes([0.81, 0.385, 0.13, 0.025]), "Fin entren.", 2012, 2017, valinit=INICIAL["fin"], valstep=1)
n_fin = fig.text(0.745, 0.36, "ultimo ano para entrenar; lo demas se predice", fontsize=8, color=GRIS)
boton = Button(fig.add_axes([0.80, 0.29, 0.14, 0.045]), "Reiniciar")

# Cuadro de ayuda: explica lo que tengas debajo del mouse
AYUDA = {
    "inicio": "QUE ESTOY VIENDO?\nEl SVR aprende con los precios reales de la zona verde y predice "
              "los anos siguientes (linea roja). La linea azul es lo que de verdad paso. "
              "Pasa el mouse sobre cualquier control y aqui te explico que hace.",
    "grafica": "GRAFICA PRINCIPAL:\n- Gris: precios reales del S&P 500.\n- Zona verde: los anos "
               "con los que el modelo aprende.\n- Azul: lo que de verdad paso en el periodo a "
               "predecir.\n- Roja: lo que predijo el SVR.",
    "errores": "ERROR DE CADA MES: que tan lejos quedo la prediccion de lo real, en porcentaje.\n"
               "- Rojo: predijo de mas.\n- Azul: predijo de menos.\nBarras chiquitas = buena "
               "prediccion.",
    "resultados": "RESULTADOS:\n- Error promedio: cuanto se equivoca en promedio (%).\n- Error en "
                  "puntos: lo mismo pero en puntos del indice.\n- R2: 1 = perfecto, 0 = igual que "
                  "adivinar el promedio, negativo = peor que eso.\n- Vectores de soporte: los "
                  "meses que definen la curva del modelo.",
    "boton": "REINICIAR: regresa todo a los valores originales (rbf, mes a mes, C=1, "
             "epsilon=0.1, ventana=12, entrena hasta 2016).",
    "kernel": "KERNEL: la forma de la curva que usa el SVR.\n- rbf: curvas suaves (el mas usado).\n"
              "- linear: una linea recta.\n- poly: un polinomio; con C alto puede dispararse "
              "(sobreajuste).",
    "modo": "MODO:\n- Mes a mes: para predecir cada mes usa el precio REAL del mes anterior.\n"
            "- A ciegas: predice todo el periodo sin ver datos reales; usa sus propias "
            "predicciones, asi que los errores se acumulan.",
    "C": "C (regularizacion): que tanto castiga los errores al entrenar.\n- Bajo (0.01): modelo "
         "simple y tolerante.\n- Alto (100): se pega mucho a los datos de entrenamiento y puede "
         "fallar con datos nuevos (sobreajuste).",
    "eps": "EPSILON: el ancho del 'tubo' de tolerancia. Los errores mas chicos que epsilon no se "
           "castigan.\n- Chico: el modelo intenta acertar casi exacto (mas vectores de soporte).\n"
           "- Grande: el modelo es mas relajado y simple.",
    "vent": "VENTANA: cuantos meses anteriores mira el modelo para predecir el siguiente.\n"
            "Ej. 12 = usa como se movio el precio en el ultimo ano.\nPoca ventana = poca "
            "informacion; mucha ventana = menos ejemplos para entrenar.",
    "fin": "FIN DEL ENTRENAMIENTO: el ultimo ano que el modelo puede ver para aprender "
           "(zona verde). Todo lo que viene despues lo tiene que predecir.\n2016 = entrena "
           "2010-2016 y predice 2017-2018 (como en el pizarron).",
    "reset": "REINICIAR: todo regreso a los valores originales (rbf, mes a mes, C=1, "
             "epsilon=0.1, ventana=12, entrena hasta 2016).",
}
import textwrap
ayuda = fig.text(0.745, 0.25, "", va="top", fontsize=8.5, wrap=False,
                 bbox=dict(boxstyle="round", facecolor="#fff8dc", edgecolor="#c8a000"))


ayuda_actual = None


def mostrar_ayuda(clave):
    global ayuda_actual
    ayuda_actual = clave
    lineas = []
    for parrafo in AYUDA[clave].split("\n"):
        lineas.extend(textwrap.wrap(parrafo, 46) or [""])
    ayuda.set_text("\n".join(lineas))


# Zonas donde, al pasar el mouse, cambia la ayuda (ejes de cada control y sus textos)
ZONAS = [
    (radio_kernel.ax, "kernel"), (t_kernel, "kernel"),
    (radio_modo.ax, "modo"), (t_modo, "modo"),
    (s_C.ax, "C"), (s_C.label, "C"), (s_C.valtext, "C"), (n_C, "C"),
    (s_eps.ax, "eps"), (s_eps.label, "eps"), (s_eps.valtext, "eps"), (n_eps, "eps"),
    (s_vent.ax, "vent"), (s_vent.label, "vent"), (s_vent.valtext, "vent"), (n_vent, "vent"),
    (s_fin.ax, "fin"), (s_fin.label, "fin"), (s_fin.valtext, "fin"), (n_fin, "fin"),
    (boton.ax, "boton"), (texto, "resultados"), (ax, "grafica"), (ax_err, "errores"),
]


def al_mover_mouse(event):
    for zona_, clave in ZONAS:
        if isinstance(zona_, Axes):            # un control o una grafica
            encontrado = zona_ is event.inaxes
        else:                                  # un texto (nombre, nota o resultados)
            encontrado = zona_.contains(event)[0]
        if encontrado:
            if clave != ayuda_actual:
                mostrar_ayuda(clave)
                fig.canvas.draw_idle()
            return


fig.canvas.mpl_connect("motion_notify_event", al_mover_mouse)


def actualizar(_=None):
    global zona, barras
    C = 10 ** s_C.val
    s_C.valtext.set_text(f"{C:.2f}")
    p = dict(kernel=radio_kernel.value_selected, C=C, epsilon=s_eps.val,
             ventana=int(s_vent.val), fin=int(s_fin.val), modo=radio_modo.value_selected)
    fechas, reales, pred, mape, mae, r2, n_sv, n_train = entrenar_y_predecir(**p)

    linea_real.set_data(fechas, reales)
    linea_pred.set_data(fechas, pred)
    corte.set_xdata([np.datetime64(f"{p['fin']}-12-01")] * 2)
    if zona is not None:
        zona.remove()
    zona = ax.axvspan(df.Fecha.min(), np.datetime64(f"{p['fin']}-12-01"), color="green", alpha=0.06)
    ax.set_title(f"Entrena: 2010-{p['fin']}   |   Predice: {p['fin'] + 1}-2018   ({p['modo']})")
    ax.relim()
    ax.autoscale_view()

    if barras is not None:
        barras.remove()
    errores = (pred - reales) / reales * 100
    colores = ["tab:red" if e > 0 else "tab:blue" for e in errores]
    barras = ax_err.bar(mdates.date2num(pd.to_datetime(fechas)), errores, width=20, color=colores)
    ax_err.set_xlim(ax.get_xlim())
    ax_err.relim()
    ax_err.autoscale_view()
    ax_err.set_title("Error de cada mes (rojo = predijo de mas, azul = de menos)", fontsize=9)

    texto.set_text(
        "RESULTADOS\n"
        f"Meses para aprender:  {n_train}\n"
        f"Meses a predecir:     {len(reales)}\n"
        f"Vectores de soporte:  {n_sv}\n"
        f"Error promedio:     {mape:6.2f} %\n"
        f"Error en puntos:    {mae:6.1f}\n"
        f"R2 (1 = perfecto):  {r2:6.3f}")
    fig.canvas.draw_idle()


def reiniciar(_):
    for s in (s_C, s_eps, s_vent, s_fin):
        s.reset()
    radio_kernel.set_active(0)
    radio_modo.set_active(0)
    mostrar_ayuda("reset")


def con_ayuda(clave):
    def manejador(valor):
        mostrar_ayuda(clave)
        actualizar()
    return manejador


s_C.on_changed(con_ayuda("C"))
s_eps.on_changed(con_ayuda("eps"))
s_vent.on_changed(con_ayuda("vent"))
s_fin.on_changed(con_ayuda("fin"))
radio_kernel.on_clicked(con_ayuda("kernel"))
radio_modo.on_clicked(con_ayuda("modo"))
boton.on_clicked(reiniciar)

mostrar_ayuda("inicio")
actualizar()
if __name__ == "__main__":
    import sys
    if "--prueba" in sys.argv:          # solo para comprobar sin abrir ventana
        fig.savefig(os.path.join(AQUI, "_prueba_interactivo.png"), dpi=90)
    else:
        plt.show()
