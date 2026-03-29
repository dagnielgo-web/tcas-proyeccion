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
#  TÍTULO
# -----------------------
col_logo1, col_titulo, col_logo2 = st.columns([1, 4, 1])

with col_logo1:
    st.image("logo_esave.png", width=80)  # Logo izquierdo

with col_titulo:
    st.markdown("<h1 style='text-align: center;'>Proyección TCAS</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: gray;'>Para la flota ATR 42</h3>", unsafe_allow_html=True)

with col_logo2:
    st.image("logo_satena.png", width=88)  # Logo derecho
# -----------------------
# INPUTS
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
#  BOTÓN
# -----------------------
if st.button("Enviar"):

    if zip_file is None:
        st.error("Debes subir un archivo ZIP")
        st.stop()

    # -----------------------
    #  DESCOMPRIMIR
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
                except Exception as e:
                    st.warning(f"Error en archivo {archivo}: {e}")

    # -----------------------
    #  LEER CSV
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

            columnas = ["TRAJ__LAT_GPS","TRAJ__LON_GPS","ALT__BARO","FLIGHT__PHASE","GMT__YEAR","GMT__HOUR"] + canales

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

                #  Fases a excluir
                fases_excluidas = ["PARKING", "FINAL APPROACH", "TAXI_OUT"]

                filas_validas = filas_evento[
                    ~filas_evento["FLIGHT__PHASE"].astype(str).str.upper().isin(fases_excluidas)
                ]

#  SOLO tomar eventos válidos (SIN fallback)
                if filas_validas.empty:
                    continue

                evento = filas_validas.loc[filas_validas["ALT__BARO"].idxmax()]

                eventos.append([
                    2000 + int(evento["GMT__YEAR"]),
                    evento["TRAJ__LAT_GPS"],
                    evento["TRAJ__LON_GPS"],
                    evento["ALT__BARO"],
                    evento["FLIGHT__PHASE"],
                    (int(evento["GMT__HOUR"]) - 5) % 24
                ])
        except:
            pass

    df_eventos = pd.DataFrame(eventos, columns=["año","lat","lon","altitud","fase","hora"])

    # -----------------------
    #  FILTRO
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
    #  EVENTOS POR AÑO
    # -----------------------
    eventos_por_año = df_eventos.groupby("año").size()
    st.write("Eventos por año")
    st.dataframe(eventos_por_año.astype(float).round(2))

    # -----------------------
    #  VUELOS Y TASAS
    # -----------------------
    vuelos_por_año = {
        2015:8505,
        2016:14807,
        2017:15070,
        2018:16629,
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

# Convertir a tasa por cada 1000 vuelos
    df_tasas["Tasa"] = df_tasas["Tasa"] * 1000

# Más decimales solo en esta tabla
    df_tasas["Tasa"] = df_tasas["Tasa"].round(4)

  #  st.write("Tasas TCAS por año")
# st.dataframe(df_tasas)

    # -----------------------
    #  MAPA ACTUAL
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
        <b>Hora (COL - UTC-5):</b> {row['hora']}:00<br>
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
    #  GRÁFICAS
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
    import plotly.express as px

# Crear DataFrame seguro para Plotly
    df_alt = df_eventos["nivel_altitud"].value_counts().reset_index()
    df_alt.columns = ["Nivel de Altitud", "Cantidad de Eventos"]

    fig_altitud = px.bar(
        df_alt,
        x="Nivel de Altitud",
        y="Cantidad de Eventos",
        color="Cantidad de Eventos",
        title="Riesgo TCAS por Altitud",
        color_continuous_scale=px.colors.sequential.Viridis
    )

    st.plotly_chart(fig_altitud, use_container_width=True)

    df_fase = df_eventos["fase"].value_counts().reset_index()
    df_fase.columns = ["Fase de Vuelo", "Cantidad de Eventos"]

    fig_fase = px.bar(
        df_fase,
        x="Fase de Vuelo",
        y="Cantidad de Eventos",
        color="Cantidad de Eventos",
        title="Eventos TCAS por Fase de Vuelo",
        color_continuous_scale=px.colors.sequential.Viridis
    )

    st.plotly_chart(fig_fase, use_container_width=True)
        # -----------------------
    #  EVENTOS POR HORA
    # -----------------------
    def clasificar_hora(h):
        if 0 <= h < 6:
            return "MADRUGADA (00-06)"
        elif 6 <= h < 12:
            return "MAÑANA (06-12)"
        elif 12 <= h < 18:
            return "TARDE (12-18)"
        else:
            return "NOCHE (18-24)"

    df_eventos["rango_hora"] = df_eventos["hora"].apply(clasificar_hora)

    df_hora = df_eventos["rango_hora"].value_counts().sort_index().reset_index()
    df_hora.columns = ["Rango Horario", "Cantidad de Eventos"]

    fig_hora = px.bar(
        df_hora,
        x="Rango Horario",
        y="Cantidad de Eventos",
        color="Cantidad de Eventos",
        title="Distribución de eventos por hora del día",
        color_continuous_scale=px.colors.sequential.Viridis
    )

    st.plotly_chart(fig_hora, use_container_width=True)




    # -----------------------
    #  PROYECCIÓN
    # -----------------------
    tasa_media = np.mean(list(tasas.values())) * 1000
    crecimiento_operacional = crecimiento_operacional / 100

    ultimo_año_vuelos = max(vuelos_por_año)
    vuelos_actuales = vuelos_por_año[ultimo_año_vuelos]

    proyeccion = []

    for i in range(1, int(años_proyeccion)+1):
        año = ultimo_año_vuelos + i
        vuelos = vuelos_actuales * (1+crecimiento_operacional)**i
        eventos_estimados = (tasa_media / 1000) * vuelos
        proyeccion.append([año,vuelos,eventos_estimados])

    df_proyeccion = pd.DataFrame(
        proyeccion,
        columns=["año","vuelos_proyectados","eventos_tcas_estimados"]
    )

# -----------------------
#  TASA PROYECTADA LINEAL
# -----------------------
    años_hist = np.array(df_tasas["Año"].values)
    tasas_hist = np.array(df_tasas["Tasa"].values)

# -----------------------
#  REGRESIÓN LINEAL
# -----------------------
    if len(años_hist) >= 2:
        pendiente, intercepto = np.polyfit(años_hist, tasas_hist, 1)
    else:
        pendiente = 0
        intercepto = tasas_hist[0] if len(tasas_hist) > 0 else 0

    tasas_proyectadas = []

    for año in df_proyeccion["año"]:
        tasa = pendiente * año + intercepto

    #  evitar tasas negativas
        if tasa < 0:
            tasa = 0

        tasas_proyectadas.append(tasa)

    df_proyeccion["tasa_tcas_por_1000_vuelos"] = np.round(tasas_proyectadas, 4)

# Redondeo general (menos preciso que tasas)
    df_proyeccion = df_proyeccion.round(2)

    #titulo de medio proyeccion
    año_proyectado_usuario = int(df_proyeccion.iloc[-1]["año"])

    st.markdown(f"<h3 style='text-align: center;'>Proyección para el año {año_proyectado_usuario}</h3>", unsafe_allow_html=True)

    st.subheader("Proyección")
    st.dataframe(df_proyeccion)
    # -----------------------
#  UNIR TASAS HISTÓRICAS + PROYECTADAS
# -----------------------

# Tomar tasas proyectadas
    df_tasas_proy = df_proyeccion[["año","tasa_tcas_por_1000_vuelos"]].copy()
    df_tasas_proy.columns = ["Año","Tasa"]

# Unir con tasas históricas
    df_tasas_total = pd.concat([df_tasas, df_tasas_proy], ignore_index=True)

# Ordenar por año
    df_tasas_total = df_tasas_total.sort_values("Año")

# Mantener formato con más decimales
    df_tasas_total["Tasa"] = df_tasas_total["Tasa"].round(4)

    st.subheader("Tasas TCAS por año (Histórico + Proyección)")
    st.dataframe(df_tasas_total)

    # -----------------------
    #  MAPA FUTURO
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
#  CONCLUSIÓN AUTOMÁTICA
# -----------------------

# Último año con datos reales
    ultimo_año_real = df_eventos["año"].max()
    eventos_ultimo_año = len(df_eventos[df_eventos["año"] == ultimo_año_real])

# Año proyectado
    año_proyectado = int(df_proyeccion.iloc[-1]["año"])
    eventos_proyectados = int(df_proyeccion.iloc[-1]["eventos_tcas_estimados"])

# Incremento %
    if eventos_ultimo_año > 0:
        incremento = ((eventos_proyectados - eventos_ultimo_año) / eventos_ultimo_año) * 100
    else:
        incremento = 0

    incremento = round(incremento, 2)

# Fases más frecuentes
    top_fases = df_eventos["fase"].value_counts().head(2).index.tolist()

    fase1 = top_fases[0] if len(top_fases) > 0 else "N/A"
    fase2 = top_fases[1] if len(top_fases) > 1 else "N/A"

# Hora más frecuente
    top_hora = df_eventos["rango_hora"].value_counts().idxmax()

# Altitud más frecuente
    top_altitud = df_eventos["nivel_altitud"].value_counts().idxmax()

# Texto final
    st.markdown(f"""
    ---
    ###  Conclusión

    Se proyecta que en el año **{año_proyectado}** se tenga un incremento de eventos TCAS a **{eventos_proyectados}** en el año, mientras que en el año **{ultimo_año_real}**, en base a datos históricos, se tiene una cantidad de **{eventos_ultimo_año} eventos TCAS RA** en la flota ATR.

    Por otro lado, se evidencia que las fases con mayor cantidad de eventos TCAS RA históricamente son **{fase1}** y **{fase2}**.

    Adicionalmente, se observa una mayor frecuencia horaria de eventos en el rango **{top_hora}**, mientras que el rango de altitud con mayor ocurrencia corresponde a **{top_altitud}**.
    """)

   
# -----------------------
# ✍️ FIRMA
# -----------------------
st.markdown(
    "<p style='text-align: right; font-size: 12px;'>Diseñado por Daniel Gonzalez</p>",
    unsafe_allow_html=True
)
