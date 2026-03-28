import streamlit as st
import pandas as pd
import zipfile
import os
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Proyección TCA's</h1>", unsafe_allow_html=True)

archivo_zip = st.file_uploader("Sube el archivo ZIP", type="zip")
anios_proyectar = st.number_input("Años a proyectar", min_value=1, max_value=10, value=3)

if st.button("Enviar"):

    if archivo_zip is not None:

        df_total = []

        with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
            for file in zip_ref.namelist():

                if file.endswith(".zip"):
                    inner_zip = zipfile.ZipFile(io.BytesIO(zip_ref.read(file)))

                    for csv_file in inner_zip.namelist():
                        df = pd.read_csv(inner_zip.open(csv_file))

                        # =========================
                        # DETECCIÓN DE EVENTO TCAS
                        # =========================
                        tcas_cols = [col for col in df.columns if "TCAS__RA" in col]
                        df["evento"] = df[tcas_cols].sum(axis=1)

                        df = df[df["evento"] > 0]

                        if df.empty:
                            continue

                        # =========================
                        # FILTRO DE FASES
                        # =========================
                        fases_invalidas = ["PARKING", "FINAL APPROACH"]
                        df = df[~df["FLIGHT__PHASE"].isin(fases_invalidas)]

                        if df.empty:
                            continue

                        # =========================
                        # CREACIÓN DE FECHA
                        # =========================
                        df["datetime"] = pd.to_datetime(
                            df["GMT__YEAR"].astype(str) + "-" +
                            df["GMT__MONTH"].astype(str) + "-" +
                            df["GMT__DAY"].astype(str) + " " +
                            df["GMT__HOUR"].astype(str) + ":00:00",
                            errors='coerce'
                        )

                        # Ajuste con segundos
                        df["datetime"] = df["datetime"] + pd.to_timedelta(df["Time"], unit='s')

                        # UTC → Colombia
                        df["datetime_col"] = df["datetime"] - pd.Timedelta(hours=5)

                        df_total.append(df.iloc[0])  # primer evento válido del vuelo

        df_final = pd.DataFrame(df_total)

        # =========================
        # VARIABLES TEMPORALES
        # =========================
        df_final["year"] = df_final["datetime_col"].dt.year
        df_final["hora"] = df_final["datetime_col"].dt.hour

        # =========================
        # EVENTOS POR AÑO
        # =========================
        eventos_anuales = df_final.groupby("year").size()

        vuelos = {2024: 19347, 2025: 20000}

        tasas = {}
        for año in eventos_anuales.index:
            if año in vuelos:
                tasas[año] = (eventos_anuales[año] / vuelos[año]) * 1000

        df_tasas = pd.DataFrame({
            "Año": list(tasas.keys()),
            "Tasa": list(tasas.values())
        })

        # =========================
        # PROYECCIÓN
        # =========================
        x = np.array(df_tasas["Año"])
        y = np.array(df_tasas["Tasa"])

        coef = np.polyfit(x, y, 1)

        futuros = []
        ultimo = max(x)

        for i in range(1, anios_proyectar + 1):
            año = ultimo + i
            tasa = coef[0]*año + coef[1]
            futuros.append((año, tasa))

        df_futuros = pd.DataFrame(futuros, columns=["Año", "Tasa"])
        df_tasas_total = pd.concat([df_tasas, df_futuros])

        # =========================
        # MAPA REAL
        # =========================
        st.subheader("Mapa de eventos reales")

        mapa = folium.Map(
            location=[df_final["TRAJ__LAT_GPS"].mean(), df_final["TRAJ__LON_GPS"].mean()],
            zoom_start=6
        )

        for _, row in df_final.iterrows():
            folium.CircleMarker(
                location=[row["TRAJ__LAT_GPS"], row["TRAJ__LON_GPS"]],
                radius=5,
                popup=f"""
                Año: {row['year']}<br>
                Fase: {row['FLIGHT__PHASE']}<br>
                Hora Colombia: {row['hora']}:00
                """,
                color="blue",
                fill=True
            ).add_to(mapa)

        st_folium(mapa, width=800)

        # =========================
        # CLUSTERS
        # =========================
        st.subheader("Mapa proyectado (clusters)")

        coords = df_final[["TRAJ__LAT_GPS", "TRAJ__LON_GPS"]].dropna()

        kmeans = KMeans(n_clusters=5, random_state=0).fit(coords)

        mapa2 = folium.Map(
            location=[coords["TRAJ__LAT_GPS"].mean(), coords["TRAJ__LON_GPS"].mean()],
            zoom_start=6
        )

        for i in range(len(coords)):
            folium.CircleMarker(
                location=[coords.iloc[i,0], coords.iloc[i,1]],
                radius=5,
                color="red",
                fill=True
            ).add_to(mapa2)

        st_folium(mapa2, width=800)

        # =========================
        # EVENTOS POR HORA
        # =========================
        st.subheader("Eventos por hora")

        eventos_hora = df_final.groupby("hora").size()

        fig, ax = plt.subplots()
        eventos_hora.plot(kind="bar", ax=ax)
        st.pyplot(fig)

        # =========================
        # ALTITUD
        # =========================
        st.subheader("Riesgo por altitud")

        alt = df_final.groupby("ALT__BARO").size()

        fig2, ax2 = plt.subplots()
        alt.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

        # =========================
        # FASE
        # =========================
        st.subheader("Eventos por fase")

        fase = df_final.groupby("FLIGHT__PHASE").size()

        fig3, ax3 = plt.subplots()
        fase.plot(kind="bar", ax=ax3)
        st.pyplot(fig3)

        # =========================
        # TABLAS
        # =========================
        st.subheader("Eventos por año")
        st.dataframe(eventos_anuales.reset_index().round(2))

        st.subheader("Tasas por cada 1000 vuelos")
        st.dataframe(df_tasas_total)

        st.markdown(
            "<p style='text-align: right; font-size: 10px;'>Diseñado por Daniel Gonzalez</p>",
            unsafe_allow_html=True
        )
