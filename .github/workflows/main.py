import gspread
import pandas as pd
from prophet import Prophet
import json
import os # Necesario para leer los secretos

print("Iniciando script de pronóstico...")

# --- PASO A: Autenticación con Service Account ---
# El script leerá las credenciales desde un "secreto" de GitHub
try:
    google_creds_json = os.environ['GOOGLE_CREDS']
    google_creds_dict = json.loads(google_creds_json)
    scoped_creds = gspread.service_account_from_dict(google_creds_dict)
    gc = gspread.authorize(scoped_creds)
    print("Autenticación con Google exitosa.")
except Exception as e:
    print(f"ERROR DE AUTENTICACIÓN: {e}")
    exit() # Detiene el script si no se puede autenticar

# --- PASO B: Configuración y Conexión ---
ID_HOJA_CALCULO = "1Y4zpiZttTocuScTeKXtYQ5LNplSgtVpE2CXeaVeRGn4"
NOMBRE_PESTAÑA = "Data_General"

# Columnas de tu hoja (letras)
COLUMNA_INICIO_DATOS = 'I'
COLUMNA_FIN_DATOS = 'S'
COLUMNA_INICIO_PRONOSTICO = 'T'

try:
    hoja_de_calculo = gc.open_by_key(ID_HOJA_CALCULO)
    hoja_ventas = hoja_de_calculo.worksheet(NOMBRE_PESTAÑA)
    print("Conexión con la hoja de cálculo exitosa.")
except Exception as e:
    print(f"ERROR DE CONEXIÓN CON LA HOJA: {e}")
    exit()

# --- PASO C: Lectura y Preparación de Datos ---
print("Leyendo y preparando los datos...")
datos = hoja_ventas.get_all_values()
df_completo = pd.DataFrame(datos)

df_completo.columns = df_completo.iloc[1]
df_ventas = df_completo.iloc[2:].reset_index(drop=True)

# Las columnas en pandas se numeran desde 0. I es 8, S es 18. El slice es [8:19].
fechas_historicas = pd.to_datetime(df_ventas.columns[8:19])

# --- PASO D: Procesamiento y Pronóstico ---
print("Iniciando el proceso de pronóstico para cada producto...")

todos_los_pronosticos = []
meses_a_pronosticar = 5 # Pronóstico para 5 meses

for index, row in df_ventas.iterrows():
    ventas_historicas_str = row[8:19]
    ventas_historicas_num = pd.to_numeric(ventas_historicas_str, errors='coerce').fillna(0)
    
    df_producto = pd.DataFrame({'ds': fechas_historicas, 'y': ventas_historicas_num.values})
    df_producto.loc[df_producto['y'] == 0, 'y'] = None

    if len(df_producto.dropna()) < 2:
        pronostico_final = [0] * meses_a_pronosticar
    else:
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df_producto)
        future = model.make_future_dataframe(periods=meses_a_pronosticar, freq='MS')
        forecast = model.predict(future)
        valores_pronosticados = forecast['yhat'].tail(meses_a_pronosticar).tolist()
        pronostico_final = [max(0, round(val)) for val in valores_pronosticados]

    todos_los_pronosticos.append(pronostico_final)
    print(f"Pronóstico para fila {index + 3} completado.")

# --- PASO E: Escritura de resultados ---
print("Escribiendo resultados en la Google Sheet...")
celda_inicio_escritura = f"{COLUMNA_INICIO_PRONOSTICO}3" 
hoja_ventas.update(celda_inicio_escritura, todos_los_pronosticos)

print("¡PROCESO COMPLETADO EXITOSAMENTE!")
