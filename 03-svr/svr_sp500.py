# SVR (Support Vector Regression / Regresion por Vectores de Soporte)
# -------------------------------------------------------------------
# SVR predice un VALOR NUMERICO (no una clase). Ejemplo del pizarron:
#   - Dataset de 2010 a 2018  -> precio mensual del indice S&P 500 (datos reales)
#   - Entrenamiento: 2010 a 2016
#   - Prueba / prediccion: 2017 a 2018
#   - Se compara lo real contra lo que predijo el modelo.
#
# Como predice: para estimar el precio de un mes, el modelo mira los precios de los
# 12 meses anteriores (ventana). Aprende esa relacion con los datos de 2010-2016 y luego
# la usa para predecir cada mes de 2017-2018.
#
#   python svr_sp500.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

AQUI = os.path.dirname(os.path.abspath(__file__))
VENTANA = 12                 # meses anteriores que usa el modelo para predecir el siguiente
FIN_ENTRENAMIENTO = "2016-12-01"

# ---------------- 1. Cargar datos ----------------
df = pd.read_csv(os.path.join(AQUI, "sp500_2010_2018.csv"), parse_dates=["Fecha"])
print(f"Datos: {len(df)} meses, de {df.Fecha.min():%Y-%m} a {df.Fecha.max():%Y-%m}")

# ---------------- 2. Armar X (12 meses anteriores) y y (mes actual) ----------------
# Se trabaja con el cambio porcentual mensual (retorno). Asi el modelo aprende "como se mueve"
# el precio y puede predecir valores mas altos que los que vio al entrenar.
precios = df.SP500.values
retornos = np.diff(precios) / precios[:-1]          # retorno del mes i+1 respecto al mes i
X, y, fechas, precio_anterior, precio_real = [], [], [], [], []
for i in range(VENTANA, len(retornos)):
    X.append(retornos[i - VENTANA:i])
    y.append(retornos[i])
    fechas.append(df.Fecha[i + 1])
    precio_anterior.append(precios[i])
    precio_real.append(precios[i + 1])
X, y = np.array(X), np.array(y)
fechas = pd.Series(fechas)
precio_anterior, precio_real = np.array(precio_anterior), np.array(precio_real)

# ---------------- 3. Separar entrenamiento (2010-2016) y prueba (2017-2018) ----------------
entrena = (fechas <= FIN_ENTRENAMIENTO).values
prueba = ~entrena
print(f"Entrenamiento: {entrena.sum()} meses ({fechas[entrena].min():%Y-%m} a {fechas[entrena].max():%Y-%m})")
print(f"Prueba:        {prueba.sum()} meses ({fechas[prueba].min():%Y-%m} a {fechas[prueba].max():%Y-%m})")

# ---------------- 4. Escalar y entrenar el SVR ----------------
escala_X = StandardScaler().fit(X[entrena])
escala_y = StandardScaler().fit(y[entrena].reshape(-1, 1))
modelo = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
modelo.fit(escala_X.transform(X[entrena]), escala_y.transform(y[entrena].reshape(-1, 1)).ravel())
print(f"Modelo entrenado: SVR(kernel='rbf'), vectores de soporte = {len(modelo.support_)}")

# ---------------- 5. Predecir 2017-2018 ----------------
ret_pred = escala_y.inverse_transform(
    modelo.predict(escala_X.transform(X[prueba])).reshape(-1, 1)).ravel()
precio_pred = precio_anterior[prueba] * (1 + ret_pred)   # de retorno predicho a precio predicho
reales = precio_real[prueba]

# ---------------- 6. Comparar real vs prediccion ----------------
print("\nComparacion mes por mes (2017-2018):")
print(f"{'Mes':<9}{'Real':>10}{'Prediccion':>12}{'Error':>10}{'Error %':>9}")
for f, r, p in zip(fechas[prueba], reales, precio_pred):
    print(f"{f:%Y-%m}  {r:>10.2f}{p:>12.2f}{p - r:>10.2f}{abs(p - r) / r * 100:>8.2f}%")

mae = mean_absolute_error(reales, precio_pred)
rmse = np.sqrt(mean_squared_error(reales, precio_pred))
mape = np.mean(np.abs((reales - precio_pred) / reales)) * 100
r2 = r2_score(reales, precio_pred)
print("\nQue tan bien predijo:")
print(f"  MAE  (error promedio en puntos)         = {mae:.2f}")
print(f"  RMSE (castiga mas los errores grandes)  = {rmse:.2f}")
print(f"  MAPE (error promedio en porcentaje)     = {mape:.2f}%")
print(f"  R2   (1 = perfecto, 0 = no sirve)       = {r2:.3f}")

# ---------------- 7. Grafica ----------------
plt.figure(figsize=(11, 5))
plt.plot(df.Fecha, df.SP500, color="gray", label="Datos reales (entrenamiento 2010-2016)")
plt.plot(fechas[prueba], reales, color="tab:blue", marker="o", markersize=4, label="Real 2017-2018")
plt.plot(fechas[prueba], precio_pred, color="tab:red", marker="x", linestyle="--",
         label="Prediccion SVR 2017-2018")
plt.axvline(pd.Timestamp(FIN_ENTRENAMIENTO), color="black", linestyle=":", label="Fin del entrenamiento")
plt.title("SVR: prediccion del S&P 500 (entrenado 2010-2016, probado 2017-2018)")
plt.xlabel("Fecha")
plt.ylabel("Precio del S&P 500")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(AQUI, "svr_resultado.png"), dpi=120)
print("\nGrafica guardada en svr_resultado.png")
plt.show()
