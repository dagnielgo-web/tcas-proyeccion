import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import json
import folium
from streamlit_folium import st_folium
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📊 Análisis y Proyección TCAS")

# -----------------------------
# CARGA DE ARCHIVO
# -----------------------------
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

                        # 🔍 VALIDAR COLUMNAS CLAVE
                        columnas_requeridas = [
                            "RA", "GMT__YEAR", "GMT__HOUR",
                            "TRAJ__LAT_GPS", "TRAJ__LON_GPS",
                            "ALT__BARO", "FLIGHT__PHASE"
                        ]

                        if not all(col in df.columns for col in columnas_requeridas):
                            continue

                        # -----------------------------
                        # FILTRO DE EVENTOS
                        # -----------------------------
                        df_validos = df[
                            (df["RA"] == 1) &
                            (~df["FLIGHT__PHASE"].isin(["PARKING", "FINAL APPROACH"]))
                        ]

                        if not df_validos.empty:

                            evento = df_validos.iloc[0]

                            año = int(evento["GMT__YEAR"])

                            # 🕒 HORA COLOMBIA (UTC-5)
                            hora_utc = int(evento["GMT__HOUR"])
                            hora_col = (hora_utc - 5) % 24

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

    # -----------------------------
    # VALIDACIÓN FINAL
    # -----------------------------
    if len(eventos) == 0:
        st.warning("No hay datos válidos en el archivo")
        st.stop()

    # -----------------------------
    # DATAFRAME
    # -----------------------------
    df_eventos = pd.DataFrame(
        eventos,
        columns=["año", "hora", "lat", "lon", "altitud", "fase"]
    )

    # -----------------------------
    # TABLA DE TASAS
    # -----------------------------
    tabla = df_eventos.groupby("año").size().reset_index(name="eventos")
    tabla["vuelos"] = total_vuelos
    tabla["Tasa_real_x1000"] = (tabla["eventos"] / tabla["vuelos"]) * 1000

    # -----------------------------
    # PROYECCIÓN
    # -----------------------------
    años_proyectar = st.number_input("Años a proyectar", 1, 10, 3)

    ultimo_año = tabla["año"].max()
    tasa_base = tabla["Tasa_real_x1000"].iloc[-1]

    factor_crecimiento = 1.05

    proyecciones = []

    for i in range(1, años_proyectar + 1):
        año_futuro = ultimo_año + i
        tasa_proy = tasa_base * (factor_crecimiento ** i)

        proyecciones.append([
            año_futuro,
            np.nan,
            np.nan,
            tasa_proy
        ])

    df_proy = pd.DataFrame(
        proyecciones,
        columns=["año", "eventos", "vuelos", "Tasa_real_x1000"]
    )

    df_tabla = pd.concat([tabla, df_proy], ignore_index=True)

    # -----------------------------
    # TABLA FINAL
    # -----------------------------
    st.subheader("📋 Tabla de tasas (por cada 1000 vuelos)")
    st.dataframe(df_tabla)

    # -----------------------------
    # GRÁFICA DE TASAS
    # -----------------------------
    st.subheader("📈 Evolución de tasas")

    fig, ax = plt.subplots()
    ax.plot(df_tabla["año"], df_tabla["Tasa_real_x1000"], marker='o')
    ax.set_xlabel("Año")
    ax.set_ylabel("Tasa x1000")
    ax.set_title("Tasa TCAS")
    st.pyplot(fig)

    # -----------------------------
    # GRÁFICA POR HORA
    # -----------------------------
    st.subheader("🕒 Distribución de eventos por hora (Colombia)")

    fig2, ax2 = plt.subplots()
    df_eventos["hora"].hist(ax=ax2, bins=24)
    ax2.set_xlabel("Hora")
    ax2.set_ylabel("Eventos")
    st.pyplot(fig2)

    # -----------------------------
    # HEATMAP REAL
    # -----------------------------
    st.subheader(f"🔥 Heatmap de eventos reales | Eventos: {len(df_eventos)}")

    mapa = folium.Map(
        location=[df_eventos["lat"].mean(), df_eventos["lon"].mean()],
        zoom_start=5
    )

    for _, row in df_eventos.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            popup=f"""
            <b>Año:</b> {row['año']}<br>
            <b>Hora (COL):</b> {row['hora']}:00<br>
            <b>Fase:</b> {row['fase']}<br>
            <b>Altitud:</b> {row['altitud']}
            """,
            color="red",
            fill=True
        ).add_to(mapa)

    st_folium(mapa, width=900)

    # -----------------------------
    # HEATMAP PROYECTADO (KDE)
    # -----------------------------
    st.subheader("🔥 Heatmap proyectado con hotspots")

    coords = np.vstack([df_eventos["lat"], df_eventos["lon"]])

    try:
        kde = gaussian_kde(coords)

        n_puntos = len(df_eventos) * años_proyectar

        nuevas_coords = kde.resample(n_puntos)

        mapa2 = folium.Map(
            location=[df_eventos["lat"].mean(), df_eventos["lon"].mean()],
            zoom_start=5
        )

        for i in range(n_puntos):
            folium.CircleMarker(
                location=[nuevas_coords[0][i], nuevas_coords[1][i]],
                radius=3,
                color="blue",
                fill=True
            ).add_to(mapa2)

        st_folium(mapa2, width=900)

    except:
        st.warning("No se pudo generar el heatmap proyectado (datos insuficientes)")
