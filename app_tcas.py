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

# -----------------------
# 🎯 TÍTULO
# -----------------------
st.markdown("<h1 style='text-align: center;'>Proyección TCAS para ATR 42</h1>", unsafe_allow_html=True)

# -----------------------
# 📁 INPUTS
# -----------------------
zip_file = st.file_uploader("Sube la carpeta comprimida (.zip)", type=["zip"])

col1, col2 = st.columns(2)

with col1:
    año_inicio = st.number_input("Año inicial", value=2019)
    crecimiento_operacional = st.number_input("Crecimiento operacional (%)", value=5.0)

with col2:
    año_fin = st.number_input("Año final", value=2024)
    años_proyeccion = st.number_input("Años a proyectar", value=3)

# -----------------------
# 🚀 BOTÓN
# -----------------------
if st.button("Enviar"):

    if zip_file is None:
        st.error("Debes subir un archivo ZIP")
        st.stop()

    # -----------------------
    # 📂 DESCOMPRIMIR
    # -----------------------
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

    # -----------------------
    # 📊 LEER CSV
    # -----------------------
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

            columnas = ["TRAJ__LAT_GPS","TRAJ__LON_GPS","ALT__BARO","FLIGHT__PHASE","GMT__YEAR"] + canales

            if not all(col in df.columns for col in columnas):
                continue

            for c in canales:
                df[c] = df[c].astype(str).str.strip().str.upper()

            # -----------------------
            # 🎯 DETECCIÓN DE EVENTOS
            # -----------------------
            mask = ~df[canales].isin(estados_normales)
            filas_evento = df[mask.any(axis=1)]

            if not filas_evento.empty:

                # 🚫 Fases a excluir
                fases_excluidas = ["PARKING", "FINAL APPROACH"]

                filas_validas = filas_evento[
                    ~filas_evento["FLIGHT__PHASE"].astype(str).str.upper().isin(fases_excluidas)
                ]

                # ✅ Tomar evento válido
                if not filas_validas.empty:
                    evento = filas_validas.iloc[0]
                else:
                    evento = filas_evento.iloc[0]

                eventos.append([
                    2000 + int(evento["GMT__YEAR"]),
                    evento["TRAJ__LAT_GPS"],
                    evento["TRAJ__LON_GPS"],
                    evento["ALT__BARO"],
                    evento["FLIGHT__PHASE"]
                ])
        except:
            pass

    df_eventos = pd.DataFrame(eventos, columns=["año","lat","lon","altitud","fase"])

    # -----------------------
    # 📉 FILTRO
    # -----------------------
    df_eventos = df_eventos[
        (df_eventos["año"] >= año_inicio) &
        (df_eventos["año"] <= año_fin)
    ].copy()

    if df_eventos.empty:
        st.error("No hay datos en ese rango")
        st.stop()

    st.subheader("Eventos detectados")
    st.write(len(df_eventos))

    # -----------------------
    # 📊 EVENTOS POR AÑO
    # -----------------------
    eventos_por_año = df_eventos.groupby("año").size()
    st.write("Eventos por año")
    st.dataframe(eventos_por_año.astype(float).round(2))

    # -----------------------
    # 📊 VUELOS Y TASAS
    # -----------------------
    vuelos_por_año = {
        2019:17235,
        2020:7331,
        2021:15737,
        2022:16477,
        2023:17630,
        2024:19347,
        2025:20256
    }

    tasas = {}

    for año in eventos_por_año.index:
        if año in vuelos_por_año:
            tasas[año] = eventos_por_año[año] / vuelos_por_año[año]

    df_tasas = pd.DataFrame(list(tasas.items()), columns=["Año","Tasa"])
    df_tasas["Tasa"] = df_tasas["Tasa"].round(2)

    st.write("Tasas TCAS por año")
    st.dataframe(df_tasas)

    # -----------------------
    # 🗺️ MAPA ACTUAL
    # -----------------------
    ultimo_año = df_eventos["año"].max()
    df_ultimo = df_eventos[df_eventos["año"] == ultimo_año]

    total_eventos = len(df_ultimo)

    mapa = folium.Map(location=[4.5, -74], zoom_start=6)

    HeatMap(
        df_ultimo[["lat","lon"]].values,
        radius=17,
        blur=20,
        max_zoom=15
    ).add_to(mapa)

    for _, row in df_ultimo.iterrows():
        info = f"""
        <b>Año:</b> {row['año']}<br>
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

    # -----------------------
    # 📈 PROYECCIÓN
    # -----------------------
    tasa_media = np.mean(list(tasas.values()))
    crecimiento_operacional = crecimiento_operacional / 100

    ultimo_año_vuelos = max(vuelos_por_año)
    vuelos_actuales = vuelos_por_año[ultimo_año_vuelos]

    proyeccion = []

    for i in range(1, int(años_proyeccion)+1):
        año = ultimo_año_vuelos + i
        vuelos = vuelos_actuales * (1+crecimiento_operacional)**i
        eventos_estimados = tasa_media * vuelos
        proyeccion.append([año,vuelos,eventos_estimados])

    df_proyeccion = pd.DataFrame(
        proyeccion,
        columns=["año","vuelos_proyectados","eventos_tcas_estimados"]
    ).round(2)

    st.subheader("Proyección")
    st.dataframe(df_proyeccion)

    # -----------------------
    # 🗺️ MAPA FUTURO
    # -----------------------
    kde = KernelDensity(bandwidth=0.03)
    kde.fit(df_eventos[["lat","lon"]].values)

    eventos_futuros = int(df_proyeccion.iloc[-1]["eventos_tcas_estimados"])

    simulados = kde.sample(eventos_futuros)
    df_simulados = pd.DataFrame(simulados, columns=["lat","lon"])

    mapa_futuro = folium.Map(location=[4.5,-74], zoom_start=6)

    df_simulados["peso"] = 1

    HeatMap(
        df_simulados[["lat","lon","peso"]].values,
        radius=17,
        blur=20,
        max_zoom=15
    ).add_to(mapa_futuro)

    df_simulados["lat_bin"] = df_simulados["lat"].round(1)
    df_simulados["lon_bin"] = df_simulados["lon"].round(1)

    zonas = df_simulados.groupby(["lat_bin","lon_bin"]).size().reset_index(name="eventos")

    for _, row in zonas.iterrows():
        folium.CircleMarker(
            location=[row["lat_bin"], row["lon_bin"]],
            radius=5 + row["eventos"] * 0.15,
            popup=f"Eventos estimados: {int(row['eventos'])}",
            color="blue",
            fill=True,
            fill_opacity=0.6
        ).add_to(mapa_futuro)

    año_proyectado = int(df_proyeccion.iloc[-1]["año"])

    titulo = f"""
    <h3 align="center">
    Proyección TCAS {año_proyectado}<br>
    Eventos estimados: {eventos_futuros}
    </h3>
    """
    mapa_futuro.get_root().html.add_child(folium.Element(titulo))

    st.subheader("Mapa proyectado")
    st.components.v1.html(mapa_futuro._repr_html_(), height=600)

    # -----------------------
    # 📊 GRÁFICAS
    # -----------------------
    def clasificar_altitud(alt):
        if alt < 10000:
            return "LOW (<10000 ft)"
        elif alt < 20000:
            return "MEDIUM (10000-20000 ft)"
        elif alt < 30000:
            return "HIGH (20000-30000 ft)"
        else:
            return "CRUISE (>30000 ft)"

    df_eventos["nivel_altitud"] = df_eventos["altitud"].apply(clasificar_altitud)

    riesgo_altitud = df_eventos["nivel_altitud"].value_counts()

    st.subheader("Riesgo TCAS por Altitud General")
    fig1, ax1 = plt.subplots()
    riesgo_altitud.plot(kind="bar", ax=ax1, title="Riesgo TCAS por Altitud")
    st.pyplot(fig1)

    riesgo_fase = df_eventos["fase"].value_counts()

    st.subheader("Eventos TCAS por Fase de Vuelo General")
    fig2, ax2 = plt.subplots()
    riesgo_fase.plot(kind="bar", ax=ax2, title="Eventos TCAS por fase de vuelo")
    st.pyplot(fig2)

# -----------------------
# ✍️ FIRMA
# -----------------------
st.markdown(
    "<p style='text-align: right; font-size: 12px;'>Diseñado por Daniel Gonzalez</p>",
    unsafe_allow_html=True
