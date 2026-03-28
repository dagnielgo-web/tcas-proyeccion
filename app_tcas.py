import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.neighbors import KernelDensity
import tempfile
import matplotlib.pyplot as plt

# =========================
# TÍTULO
# =========================
st.markdown("<h1 style='text-align: center;'>Proyección TCAS para ATR 42</h1>", unsafe_allow_html=True)

# =========================
# INPUTS
# =========================
zip_file = st.file_uploader("Sube la carpeta comprimida (.zip)", type=["zip"])

col1, col2 = st.columns(2)

with col1:
    año_inicio = st.number_input("Año inicial", value=2019)
    crecimiento_operacional = st.number_input("Crecimiento operacional (%)", value=5.0)

with col2:
    año_fin = st.number_input("Año final", value=2024)
    años_proyeccion = st.number_input("Años a proyectar", value=3)

# =========================
# BOTÓN
# =========================
if st.button("Enviar"):

    if zip_file is None:
        st.error("Debes subir un archivo ZIP")
        st.stop()

    # =========================
    # DESCOMPRESIÓN
    # =========================
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".zip"):
                try:
                    with zipfile.ZipFile(os.path.join(root,file),'r') as z:
                        z.extractall(root)
                except:
                    pass

    # =========================
    # LECTURA CSV
    # =========================
    csv_files = []

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root,file))

    eventos = []

    canales = ["TCAS__RA_1","TCAS__RA_2","TCAS__RA_3","TCAS__RA_4"]
    estados_normales = ["NO ADVISORY","NRD"]

    for archivo in csv_files:
        try:
            df = pd.read_csv(archivo)

            columnas = ["TRAJ__LAT_GPS","TRAJ__LON_GPS","ALT__BARO","FLIGHT__PHASE","GMT__YEAR","GMT__HOUR"] + canales

            if not all(col in df.columns for col in columnas):
                continue

            for c in canales:
                df[c] = df[c].astype(str).str.strip().str.upper()

            mask = ~df[canales].isin(estados_normales)
            filas_evento = df[mask.any(axis=1)]

            if not filas_evento.empty:

                # 🔥 FASES EXCLUIDAS (incluye TAXI_OUT)
                fases_excluidas = ["PARKING", "FINAL APPROACH", "TAXI_OUT"]

                filas_validas = filas_evento[
                    ~filas_evento["FLIGHT__PHASE"].astype(str).str.upper().isin(fases_excluidas)
                ]

                if not filas_validas.empty:
                    evento = filas_validas.iloc[0]
                else:
                    evento = filas_evento.iloc[0]

                # 🔥 HORA COLOMBIA
                hora_col = (int(evento["GMT__HOUR"]) - 5) % 24

                eventos.append([
                    2000 + int(evento["GMT__YEAR"]),
                    hora_col,
                    evento["TRAJ__LAT_GPS"],
                    evento["TRAJ__LON_GPS"],
                    evento["ALT__BARO"],
                    evento["FLIGHT__PHASE"]
                ])
        except:
            pass

    df_eventos = pd.DataFrame(eventos, columns=["año","hora","lat","lon","altitud","fase"])

    # =========================
    # FILTRO POR AÑO
    # =========================
    df_eventos = df_eventos[
        (df_eventos["año"] >= año_inicio) &
        (df_eventos["año"] <= año_fin)
    ].copy()

    if df_eventos.empty:
        st.error("No hay datos en ese rango")
        st.stop()

    # =========================
    # MÉTRICAS
    # =========================
    st.subheader("Eventos detectados")
    st.write(len(df_eventos))

    eventos_por_año = df_eventos.groupby("año").size()
    st.write("Eventos por año")
    st.dataframe(eventos_por_año)

    # =========================
    # VUELOS
    # =========================
    vuelos_por_año = {
        2019:17235,2020:7331,2021:15737,2022:16477,
        2023:17630,2024:19347,2025:20256
    }

    # =========================
    # TASAS (x1000 vuelos)
    # =========================
    tasas = {}

    for año in eventos_por_año.index:
        if año in vuelos_por_año:
            tasas[año] = (eventos_por_año[año] / vuelos_por_año[año]) * 1000

    df_tasas = pd.DataFrame(list(tasas.items()), columns=["Año","Tasa"])

    # =========================
    # REGRESIÓN LINEAL
    # =========================
    x = df_tasas["Año"]
    y = df_tasas["Tasa"]

    coef = np.polyfit(x, y, 1)

    años_futuros = [max(x) + i for i in range(1, int(años_proyeccion)+1)]
    tasas_futuras = [coef[0]*a + coef[1] for a in años_futuros]

    df_futuro = pd.DataFrame({
        "Año": años_futuros,
        "Tasa": tasas_futuras
    })

    df_tasas = pd.concat([df_tasas, df_futuro], ignore_index=True)

    st.write("Tasas TCAS por año (x1000 vuelos)")
    st.dataframe(df_tasas)

    # =========================
    # GRÁFICA POR HORA
    # =========================
    st.subheader("Distribución de eventos por hora")

    fig_h, ax_h = plt.subplots()
    df_eventos["hora"].hist(bins=24, ax=ax_h)

    ax_h.set_xlabel("Hora del día (COL)")
    ax_h.set_ylabel("Número de eventos")
    ax_h.set_title("Eventos TCAS por hora")

    st.pyplot(fig_h)

    # =========================
    # MAPA (AÑO MÁS RECIENTE)
    # =========================
    ultimo_año = df_eventos["año"].max()
    df_ultimo = df_eventos[df_eventos["año"] == ultimo_año]

    total_eventos = len(df_ultimo)

    mapa = folium.Map(location=[4.5, -74], zoom_start=6)

    HeatMap(df_ultimo[["lat","lon"]].values, radius=17, blur=20).add_to(mapa)

    for _, row in df_ultimo.iterrows():

        info = f"""
        <b>Año:</b> {row['año']}<br>
        <b>Hora (COL):</b> {row['hora']}<br>
        <b>Altitud:</b> {round(row['altitud'],2)} ft<br>
        <b>Lat:</b> {round(row['lat'],5)}<br>
        <b>Lon:</b> {round(row['lon'],5)}
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="red",
            fill=True,
            fill_opacity=0.3,
            popup=folium.Popup(info, max_width=300)
        ).add_to(mapa)

    titulo = f"""
    <h3 align="center">
    Eventos TCAS {ultimo_año}<br>
    Eventos reales detectados: {total_eventos}
    </h3>
    """

    mapa.get_root().html.add_child(folium.Element(titulo))

    st.subheader("Mapa actual")
    st.components.v1.html(mapa._repr_html_(), height=600)

    # =========================
    # (NO SE MODIFICA NADA MÁS)
    # =========================
