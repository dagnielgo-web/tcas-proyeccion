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

st.markdown("<h1 style='text-align: center;'>Proyección TCA's</h1>", unsafe_allow_html=True)

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

                evento = filas_evento.iloc[0]

                year_raw = int(evento["GMT__YEAR"])
                año = 2000 + year_raw if year_raw < 100 else year_raw

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

    # =========================
    # 📊 EVENTOS POR AÑO
    # =========================
    eventos_por_año = df_eventos.groupby("año").size()

    vuelos_por_año = {
        2019:17235, 2020:7331, 2021:15737,
        2022:16477, 2023:17630, 2024:19347, 2025:20256
    }

    tasas = {}
    for año in eventos_por_año.index:
        if año in vuelos_por_año:
            tasas[año] = (eventos_por_año[año] / vuelos_por_año[año]) * 1000

    # =========================
    # 🗺️ HEATMAP ACTUAL
    # =========================
    ultimo_año = df_eventos["año"].max()
    df_ultimo = df_eventos[df_eventos["año"] == ultimo_año]

    mapa = folium.Map(location=[4.5, -74], zoom_start=6)
    HeatMap(df_ultimo[["lat","lon"]].values).add_to(mapa)

    st.subheader(f"Heatmap año {ultimo_año} | Eventos: {len(df_ultimo)}")
    st.components.v1.html(mapa._repr_html_(), height=600)

    # =========================
    # 📈 PROYECCIÓN
    # =========================
    factor_crecimiento = 1 + (crecimiento_operacional / 100)
    eventos_base = len(df_ultimo)

    proyecciones = []
    for i in range(1, int(años_proyeccion)+1):
        año_proy = ultimo_año + i
        eventos_proy = int(eventos_base * (factor_crecimiento ** i))
        proyecciones.append([año_proy, eventos_proy])

    df_proy = pd.DataFrame(proyecciones, columns=["Año","Eventos_proyectados"])

    # =========================
    # 🔥 HEATMAP PROYECTADO
    # =========================
    coords = df_ultimo[["lat","lon"]].values
    kde = KernelDensity(bandwidth=0.5).fit(coords)

    n_puntos = df_proy["Eventos_proyectados"].iloc[-1]

    lat_sim = np.random.uniform(df_ultimo["lat"].min(), df_ultimo["lat"].max(), n_puntos)
    lon_sim = np.random.uniform(df_ultimo["lon"].min(), df_ultimo["lon"].max(), n_puntos)

    coords_sim = np.vstack([lat_sim, lon_sim]).T
    densidad = kde.score_samples(coords_sim)

    coords_hot = coords_sim[densidad > np.percentile(densidad, 75)]

    mapa_proy = folium.Map(location=[4.5, -74], zoom_start=6)
    HeatMap(coords_hot).add_to(mapa_proy)

    st.subheader(f"Heatmap proyectado año {df_proy['Año'].iloc[-1]} | Eventos proyectados: {n_puntos}")
    st.components.v1.html(mapa_proy._repr_html_(), height=600)

    # =========================
    # 📊 TABLA COMPLETA
    # =========================
    tabla = []

    for año in eventos_por_año.index:
        tabla.append([
            año,
            eventos_por_año[año],
            vuelos_por_año.get(año, np.nan),
            tasas.get(año, np.nan),
            np.nan
        ])

    tasa_base = list(tasas.values())[-1]

    for i, row in df_proy.iterrows():
        tabla.append([
            row["Año"],
            row["Eventos_proyectados"],
            np.nan,
            np.nan,
            tasa_base * (factor_crecimiento ** (i+1))
        ])

    df_tabla = pd.DataFrame(tabla, columns=[
        "Año","Eventos","Vuelos","Tasa_real_x1000","Tasa_proyectada_x1000"
    ]).sort_values("Año")

    st.subheader("Tasas TCAS por cada 1000 vuelos (Histórico + Proyección)")
    st.dataframe(df_tabla.round(3))

    # =========================
    # 📊 GRÁFICAS ORIGINALES
    # =========================

    st.subheader("Eventos por hora (Colombia)")
    df_eventos.groupby("hora").size().plot(kind="bar")
    st.pyplot(plt.gcf())

    st.subheader("Eventos por fase")
    df_eventos["fase"].value_counts().plot(kind="bar")
    st.pyplot(plt.gcf())

st.markdown("<p style='text-align: right;'>Diseñado por Daniel Gonzalez</p>", unsafe_allow_html=True)
