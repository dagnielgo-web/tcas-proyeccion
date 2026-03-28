import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import json
import folium
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 🔥 INTENTO DE IMPORT SEGURO
try:
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except:
    FOLIUM_OK = False

st.set_page_config(layout="wide")

st.title("📊 Análisis y Proyección TCAS")

archivo_zip = st.file_uploader("Sube el archivo ZIP de vuelos", type=["zip"])

if archivo_zip:

    eventos = []
    total_vuelos = 0

    with zipfile.ZipFile(archivo_zip, 'r') as z:
        archivos = z.namelist()

        for archivo in archivos:
            if archivo.endswith(".json"):
                total_vuelos += 1

                try:
                    with z.open(archivo) as f:
                        data = json.load(f)
                        df = pd.DataFrame(data)

                        columnas = [
                            "RA", "GMT__YEAR", "GMT__HOUR",
                            "TRAJ__LAT_GPS", "TRAJ__LON_GPS",
                            "ALT__BARO", "FLIGHT__PHASE"
                        ]

                        if not all(c in df.columns for c in columnas):
                            continue

                        df_validos = df[
                            (df["RA"] == 1) &
                            (~df["FLIGHT__PHASE"].isin(["PARKING", "FINAL APPROACH"]))
                        ]

                        if not df_validos.empty:
                            evento = df_validos.iloc[0]

                            año = int(evento["GMT__YEAR"])

                            # 🕒 HORA COLOMBIA
                            hora_col = (int(evento["GMT__HOUR"]) - 5) % 24

                            eventos.append([
                                año,
                                hora_col,
                                float(evento["TRAJ__LAT_GPS"]),
                                float(evento["TRAJ__LON_GPS"]),
                                float(evento["ALT__BARO"]),
                                str(evento["FLIGHT__PHASE"])
                            ])
                except:
                    continue

    if len(eventos) == 0:
        st.warning("No hay datos válidos en el archivo")
        st.stop()

    df_eventos = pd.DataFrame(
        eventos,
        columns=["año", "hora", "lat", "lon", "altitud", "fase"]
    )

    # -----------------------------
    # TABLA
    # -----------------------------
    tabla = df_eventos.groupby("año").size().reset_index(name="eventos")
    tabla["vuelos"] = total_vuelos
    tabla["Tasa_real_x1000"] = (tabla["eventos"] / tabla["vuelos"]) * 1000

    años_proyectar = st.number_input("Años a proyectar", 1, 10, 3)

    ultimo_año = tabla["año"].max()
    tasa_base = tabla["Tasa_real_x1000"].iloc[-1]

    factor = 1.05

    proy = []
    for i in range(1, años_proyectar + 1):
        proy.append([
            ultimo_año + i,
            np.nan,
            np.nan,
            tasa_base * (factor ** i)
        ])

    df_proy = pd.DataFrame(proy, columns=tabla.columns)
    df_tabla = pd.concat([tabla, df_proy], ignore_index=True)

    st.subheader("📋 Tabla de tasas")
    st.dataframe(df_tabla)

    # -----------------------------
    # GRÁFICA
    # -----------------------------
    fig, ax = plt.subplots()
    ax.plot(df_tabla["año"], df_tabla["Tasa_real_x1000"], marker='o')
    st.pyplot(fig)

    # -----------------------------
    # HEATMAP REAL
    # -----------------------------
    st.subheader("🔥 Heatmap real")

    mapa = folium.Map(
        location=[df_eventos["lat"].mean(), df_eventos["lon"].mean()],
        zoom_start=5
    )

    for _, row in df_eventos.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            popup=f"""
            Año: {row['año']}<br>
            Hora: {row['hora']}<br>
            Fase: {row['fase']}
            """,
            color="red",
            fill=True
        ).add_to(mapa)

    # 🔥 MOSTRAR MAPA (CON O SIN LIBRERÍA)
    if FOLIUM_OK:
        st_folium(mapa, width=900)
    else:
        st.warning("streamlit-folium no instalado → mostrando versión básica")
        st.components.v1.html(mapa._repr_html_(), height=600)

    # -----------------------------
    # HEATMAP PROYECTADO
    # -----------------------------
    st.subheader("🔥 Heatmap proyectado")

    try:
        coords = np.vstack([df_eventos["lat"], df_eventos["lon"]])
        kde = gaussian_kde(coords)

        n = len(df_eventos) * años_proyectar
        nuevas = kde.resample(n)

        mapa2 = folium.Map(
            location=[df_eventos["lat"].mean(), df_eventos["lon"].mean()],
            zoom_start=5
        )

        for i in range(n):
            folium.CircleMarker(
                location=[nuevas[0][i], nuevas[1][i]],
                radius=3,
                color="blue",
                fill=True
            ).add_to(mapa2)

        if FOLIUM_OK:
            st_folium(mapa2, width=900)
        else:
            st.components.v1.html(mapa2._repr_html_(), height=600)

    except:
        st.warning("No se pudo generar proyección")
