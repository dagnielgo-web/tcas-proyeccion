import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import json
import folium
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 🔥 PROTECCIÓN SIN CAMBIAR TU UI
try:
    from streamlit_folium import st_folium
except:
    st_folium = None

st.set_page_config(layout="wide")

st.title("📊 Análisis TCAS")

archivo_zip = st.file_uploader("Sube el ZIP", type=["zip"])

if archivo_zip:

    eventos = []
    total_vuelos = 0

    with zipfile.ZipFile(archivo_zip, 'r') as z:

        for archivo in z.namelist():

            if archivo.endswith(".json"):
                total_vuelos += 1

                try:
                    with z.open(archivo) as f:
                        data = json.load(f)
                        df = pd.DataFrame(data)

                        # 🔥 VALIDACIÓN SUAVE (NO ROMPE)
                        if "RA" not in df.columns:
                            continue

                        df_ra = df[df["RA"] == 1]

                        if df_ra.empty:
                            continue

                        # 🔥 NUEVO: FILTRO INTELIGENTE DE FASE
                        df_validos = df_ra[
                            ~df_ra["FLIGHT__PHASE"].isin(["PARKING", "FINAL APPROACH"])
                        ]

                        if not df_validos.empty:
                            evento = df_validos.iloc[0]
                        else:
                            evento = df_ra.iloc[0]  # fallback original

                        # 🔥 NUEVO: HORA COLOMBIA
                        hora_col = (int(evento["GMT__HOUR"]) - 5) % 24

                        eventos.append([
                            int(evento["GMT__YEAR"]),
                            hora_col,  # 🔥 NUEVO
                            evento["TRAJ__LAT_GPS"],
                            evento["TRAJ__LON_GPS"],
                            evento["ALT__BARO"],
                            evento["FLIGHT__PHASE"]
                        ])

                except:
                    continue

    if len(eventos) == 0:
        st.warning("No hay datos en ese rango")
        st.stop()

    # 🔥 NUEVO: columna hora agregada
    df_eventos = pd.DataFrame(
        eventos,
        columns=["año", "hora", "lat", "lon", "altitud", "fase"]
    )

    # -----------------------------
    # TABLA ORIGINAL (RESPETADA)
    # -----------------------------
    tabla = df_eventos.groupby("año").size().reset_index(name="eventos")
    tabla["vuelos"] = total_vuelos
    tabla["Tasa_real_x1000"] = (tabla["eventos"] / tabla["vuelos"]) * 1000

    # 🔥 NUEVO: columna proyectada
    tabla["Tasa_proyectada_x1000"] = np.nan

    años_proy = st.number_input("Años a proyectar", 1, 10, 3)

    ultimo_año = tabla["año"].max()
    tasa_base = tabla["Tasa_real_x1000"].iloc[-1]

    factor = 1.05

    for i in range(1, años_proy + 1):
        año_futuro = ultimo_año + i
        tasa_proy = tasa_base * (factor ** i)

        tabla = pd.concat([
            tabla,
            pd.DataFrame([{
                "año": año_futuro,
                "eventos": np.nan,
                "vuelos": np.nan,
                "Tasa_real_x1000": np.nan,
                "Tasa_proyectada_x1000": tasa_proy
            }])
        ], ignore_index=True)

    st.subheader("Tabla de tasas")
    st.dataframe(tabla)

    # -----------------------------
    # GRÁFICA ORIGINAL
    # -----------------------------
    fig, ax = plt.subplots()
    ax.plot(tabla["año"], tabla["Tasa_real_x1000"], marker='o')
    st.pyplot(fig)

    # 🔥 NUEVO: gráfica por hora
    st.subheader("Eventos por hora")
    fig2, ax2 = plt.subplots()
    df_eventos["hora"].hist(bins=24, ax=ax2)
    st.pyplot(fig2)

    # -----------------------------
    # HEATMAP ORIGINAL (RESPETADO)
    # -----------------------------
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
            Hora (COL): {row['hora']}<br>  <!-- 🔥 NUEVO -->
            Fase: {row['fase']}<br>
            Altitud: {row['altitud']}
            """,
            color="red",
            fill=True
        ).add_to(mapa)

    if st_folium:
        st_folium(mapa, width=900)
    else:
        st.components.v1.html(mapa._repr_html_(), height=600)

    # -----------------------------
    # HEATMAP PROYECTADO ORIGINAL
    # -----------------------------
    coords = np.vstack([df_eventos["lat"], df_eventos["lon"]])
    kde = gaussian_kde(coords)

    n = len(df_eventos) * años_proy
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

    if st_folium:
        st_folium(mapa2, width=900)
    else:
        st.components.v1.html(mapa2._repr_html_(), height=600)
