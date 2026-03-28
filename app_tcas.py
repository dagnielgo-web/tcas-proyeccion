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

st.markdown("<h1 style='text-align: center;'>Proyección TCAS (Traffic Collision Avoidance System)</h1>", unsafe_allow_html=True)

zip_file = st.file_uploader("Sube la carpeta comprimida (.zip)", type=["zip"])

col1, col2 = st.columns(2)

with col1:
    año_inicio = st.number_input("Año inicial", value=2019)
    crecimiento_operacional = st.number_input("Crecimiento operacional (%)", value=5.0)

with col2:
    año_fin = st.number_input("Año final", value=2024)
    años_proyeccion = st.number_input("Años a proyectar", value=3)

if st.button("Enviar"):

    if zip_file is None:
        st.error("Debes subir un archivo ZIP")
        st.stop()

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

            columnas = ["TRAJ__LAT_GPS","TRAJ__LON_GPS","ALT__BARO","FLIGHT__PHASE",
                        "GMT__YEAR","GMT__HOUR"] + canales

            if not all(col in df.columns for col in columnas):
                continue

            for c in canales:
                df[c] = df[c].astype(str).str.strip().str.upper()

            mask = ~df[canales].isin(estados_normales)
            filas_evento = df[mask.any(axis=1)]

            if not filas_evento.empty:

                fases_excluidas = ["PARKING", "FINAL APPROACH"]

                filas_validas = filas_evento[
                    ~filas_evento["FLIGHT__PHASE"].astype(str).str.upper().isin(fases_excluidas)
                ]

                if not filas_validas.empty:
                    evento = filas_validas.iloc[0]
                else:
                    evento = filas_evento.iloc[0]

                # ✅ AÑO CORRECTO
                year_raw = int(evento["GMT__YEAR"])
                if year_raw < 100:
                    año = 2000 + year_raw
                else:
                    año = year_raw

                # ✅ HORA INDEPENDIENTE (UTC → Colombia)
                hora_utc = int(evento["GMT__HOUR"])
                hora_col = (hora_utc - 5) % 24

                eventos.append([
                    año,
                    hora_col,
                    evento["TRAJ__LAT_GPS"],
                    evento["TRAJ__LON_GPS"],
                    evento["ALT__BARO"],
                    evento["FLIGHT__PHASE"]
                ])
        except:
            pass

    df_eventos = pd.DataFrame(eventos, columns=["año","hora","lat","lon","altitud","fase"])

    df_eventos = df_eventos[
        (df_eventos["año"] >= año_inicio) &
        (df_eventos["año"] <= año_fin)
    ].copy()

    if df_eventos.empty:
        st.error("No hay datos en ese rango")
        st.stop()

    st.subheader("Eventos detectados")
    st.write(len(df_eventos))

    eventos_por_año = df_eventos.groupby("año").size()
    st.write("Eventos por año")
    st.dataframe(eventos_por_año.astype(float).round(2))

    vuelos_por_año = {
        2019:17235, 2020:7331, 2021:15737,
        2022:16477, 2023:17630, 2024:19347, 2025:20256
    }

    tasas = {}

    for año in eventos_por_año.index:
        if año in vuelos_por_año:
            tasas[año] = (eventos_por_año[año] / vuelos_por_año[año]) * 1000

    df_tasas = pd.DataFrame(list(tasas.items()), columns=["Año","Tasa"])

    st.write("Tasas TCAS por año (por cada 1000 vuelos)")
    st.dataframe(df_tasas)

    # 🗺️ MAPA
    ultimo_año = df_eventos["año"].max()
    df_ultimo = df_eventos[df_eventos["año"] == ultimo_año]

    mapa = folium.Map(location=[4.5, -74], zoom_start=6)

    HeatMap(df_ultimo[["lat","lon"]].values).add_to(mapa)

    for _, row in df_ultimo.iterrows():
        info = f"""
        <b>Año:</b> {row['año']}<br>
        <b>Fase:</b> {row['fase']}<br>
        <b>Altitud:</b> {round(row['altitud'],2)} ft<br>
        <b>Hora Colombia:</b> {row['hora']}:00<br>
        <b>Lat:</b> {round(row['lat'],5)}<br>
        <b>Lon:</b> {round(row['lon'],5)}
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="red",
            fill=True,
            popup=folium.Popup(info, max_width=300)
        ).add_to(mapa)

    st.subheader("Mapa actual")
    st.components.v1.html(mapa._repr_html_(), height=600)

    # 📊 GRÁFICA POR HORA
    st.subheader("Eventos por hora (Colombia)")
    eventos_hora = df_eventos.groupby("hora").size()

    fig_hora, ax_hora = plt.subplots()
    eventos_hora.plot(kind="bar", ax=ax_hora)
    st.pyplot(fig_hora)

    # 📊 ALTITUD
    def clasificar_altitud(alt):
        if alt < 10000:
            return "LOW"
        elif alt < 20000:
            return "MEDIUM"
        elif alt < 30000:
            return "HIGH"
        else:
            return "CRUISE"

    df_eventos["nivel_altitud"] = df_eventos["altitud"].apply(clasificar_altitud)

    riesgo_altitud = df_eventos["nivel_altitud"].value_counts()

    st.subheader("Riesgo por altitud")
    fig1, ax1 = plt.subplots()
    riesgo_altitud.plot(kind="bar", ax=ax1)
    st.pyplot(fig1)

    riesgo_fase = df_eventos["fase"].value_counts()

    st.subheader("Eventos por fase")
    fig2, ax2 = plt.subplots()
    riesgo_fase.plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

st.markdown(
    "<p style='text-align: right; font-size: 12px;'>Diseñado por Daniel Gonzalez</p>",
    unsafe_allow_html=True
)
