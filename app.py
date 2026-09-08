import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN Y ESTILOS DE LA PLATAFORMA
# ==============================================================================
st.set_page_config(
    page_title="SBS Quant Lab - Logit Master v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f8fafc;
        padding: 16px;
        border-radius: 8px;
        border-left: 5px solid #1e3a8a;
        margin-bottom: 12px;
    }
    .top15-box {
        background-color: #f0fdf4;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #16a34a;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTES Y PARÁMETROS DEL MODELO CUANTITATIVO
# ==============================================================================
EXCLUIR_TICKERS = [
    'BTC-USD', 'ETH-USD', 'CONL', 'SOXL', 'LABU', 'MARA', 'CLSK', 'CIFR', 
    'IREN', 'RGTI', 'QBTS', 'POET', 'OPEN', 'BETR', 'MVST', 'UAMY', 'TMQ', 
    'SMR', 'LTBR'
]

UMBRAL_MARGEN_OP_MIN  = 12.0   # Mínimo 12% de Margen Operativo
UMBRAL_CREC_VENTAS_MIN = 5.0   # Mínimo 5% de Crecimiento en Ventas
UMBRAL_CREC_EPS_MIN    = 0.0   # No contracción de utilidades
UMBRAL_UPSIDE_MIN      = 5.0   # Mínimo 5% de upside según consenso
PRECIO_MINIMO          = 15.0  # Filtro anti-penny stocks

# Ponderaciones de la Regresión Logística Multivariada
BETA_0 =  0.20   # Intercepto
BETA_1 = -1.50   # Factor PEG Ratio (Menor es mejor)
BETA_2 =  1.20   # Factor Margen Operativo % (Mayor es mejor)
BETA_3 =  1.10   # Factor Crecimiento Ventas % (Mayor es mejor)
BETA_4 = -0.90   # Factor Descuento vs Máximo % (Mayor caída relativa suma)
BETA_5 =  1.30   # Factor Upside Wall Street % (Mayor es mejor)

# ==============================================================================
# MOTOR MATEMÁTICO: INGESTA Y PROCESAMIENTO
# ==============================================================================
def buscar_archivo_export():
    """Localiza el archivo de exportación más reciente generado por el Tablero."""
    archivos = glob.glob("*export*.csv")
    if not archivos:
        return None
    archivos.sort(key=os.path.getmtime, reverse=True)
    return archivos[0]

@st.cache_data(ttl=1800)
def procesar_modelo_logit(ruta_csv):
    """Procesa las 2 etapas del modelo cuantitativo: filtros sanitarios y regresión."""
    if not ruta_csv or not os.path.exists(ruta_csv):
        return None, None

    df_raw = pd.read_csv(ruta_csv)

    # 1. Filtro Sanitario de Exclusión (Anti-Value Traps)
    vetadas = []
    aprobadas_idx = []

    for idx, row in df_raw.iterrows():
        ticker = str(row.get('Ticker', ''))
        
        if ticker in EXCLUIR_TICKERS:
            vetadas.append({"Ticker": ticker, "Nombre": row.get('Nombre', ''), "Razón": "Activo volátil / Cripto / Apalancado"})
            continue
        if row.get('Precio_Actual', 0) < PRECIO_MINIMO:
            vetadas.append({"Ticker": ticker, "Nombre": row.get('Nombre', ''), "Razón": f"Precio bajo (<${PRECIO_MINIMO})"})
            continue
        if row.get('Margen_Op_%', 0) < UMBRAL_MARGEN_OP_MIN:
            vetadas.append({"Ticker": ticker, "Nombre": row.get('Nombre', ''), "Razón": f"Margen Operativo insuficiente ({row.get('Margen_Op_%', 0):.1f}%)"})
            continue
        if row.get('Crec_Ventas_%', 0) < UMBRAL_CREC_VENTAS_MIN:
            vetadas.append({"Ticker": ticker, "Nombre": row.get('Nombre', ''), "Razón": f"Ventas sin expansión ({row.get('Crec_Ventas_%', 0):.1f}%)"})
            continue
        if row.get('Crec_EPS_%', 0) < UMBRAL_CREC_EPS_MIN:
            vetadas.append({"Ticker": ticker, "Nombre": row.get('Nombre', ''), "Razón": f"Contracción en EPS ({row.get('Crec_EPS_%', 0):.1f}%)"})
            continue
        if row.get('Upside_B5_%', 0) < UMBRAL_UPSIDE_MIN:
            vetadas.append({"Ticker": ticker, "Nombre": row.get('Nombre', ''), "Razón": f"Upside consenso bajo ({row.get('Upside_B5_%', 0):.1f}%)"})
            continue
        if pd.isna(row.get('PEG_Ratio')) or row.get('PEG_Ratio', 0) <= 0:
            vetadas.append({"Ticker": ticker, "Nombre": row.get('Nombre', ''), "Razón": "PEG Ratio no disponible o distorsionado"})
            continue

        aprobadas_idx.append(idx)

    df_base = df_raw.loc[aprobadas_idx].copy()
    if df_base.empty:
        return pd.DataFrame(), pd.DataFrame(vetadas)

    # 2. Normalización Vectorial (Z-Scores del Universo Aprobado)
    cols_modelo = ['PEG_Ratio', 'Margen_Op_%', 'Crec_Ventas_%', 'Dif_%_vs_Max', 'Upside_B5_%']
    for c in cols_modelo:
        media = df_base[c].mean()
        desvest = df_base[c].std() if df_base[c].std() > 0 else 1.0
        df_base[f'z_{c}'] = (df_base[c] - media) / desvest

    # 3. Función Sigmoide Multivariada Logit
    df_base['score_z'] = (
        BETA_0 +
        BETA_1 * df_base['z_PEG_Ratio'] +
        BETA_2 * df_base['z_Margen_Op_%'] +
        BETA_3 * df_base['z_Crec_Ventas_%'] +
        BETA_4 * df_base['z_Dif_%_vs_Max'] +
        BETA_5 * df_base['z_Upside_B5_%']
    )
    df_base['Probabilidad_Logit'] = 1.0 / (1.0 + np.exp(-df_base['score_z']))

    # Clasificación por Convicción
    condiciones = [
        (df_base['Probabilidad_Logit'] >= 0.85),
        (df_base['Probabilidad_Logit'] >= 0.70),
        (df_base['Probabilidad_Logit'] >= 0.55),
        (df_base['Probabilidad_Logit'] >= 0.40)
    ]
    etiquetas = ["💎 GANGA INSTITUCIONAL", "🟢 Muy Fuerte", "🟡 Atractiva", "⚪ Neutral"]
    df_base['Clasificacion'] = np.select(condiciones, etiquetas, default="🔴 Descartar")

    df_aprobadas = df_base.sort_values(by=['Probabilidad_Logit', 'Margen_Op_%'], ascending=[False, False]).reset_index(drop=True)
    df_vetadas = pd.DataFrame(vetadas)

    # Guardar matriz consolidada para consumo de API o Git
    df_aprobadas.to_csv("logit_data.csv", index=False)
    df_vetadas.to_csv("vetados_quality.csv", index=False)

    return df_aprobadas, df_vetadas

# ==============================================================================
# INTERFAZ GRÁFICA DE USUARIO
# ==============================================================================
st.title("🧠 SBS Center - Laboratorio Logit Tablero v3.0")
st.caption("Motor de Asimetría Fundamental a 30 Días | Filtro Anti-Value Traps Integrado")
st.markdown("---")

archivo_activo = buscar_archivo_export()

with st.sidebar:
    st.header("⚙️ Origen de Datos")
    if archivo_activo:
        st.success(f"Archivo cargado:\n`{archivo_activo}`")
        st.write(f"Última sync: {datetime.fromtimestamp(os.path.getmtime(archivo_activo)).strftime('%Y-%m-%d %H:%M')}")
    else:
        st.error("No se localizó ningún archivo `export.csv` en el directorio.")

    if st.button("🔄 Forzar Recálculo", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("""
    **Parámetros de Entrada:**
    * Margen Op. $\ge 12\%$
    * Crec. Ventas $\ge 5\%$
    * Horizonte: **30 Días**
    * TP Anticipado: **+8% a +10%**
    """)

if archivo_activo:
    df_aprobadas, df_vetadas = procesar_modelo_logit(archivo_activo)

    if df_aprobadas is not None and not df_aprobadas.empty:
        # CONTENEDOR PRINCIPAL: TOP 15 OFICIAL A 30 DÍAS
        st.markdown("<div class='top15-box'>", unsafe_allow_html=True)
        st.subheader("🔥 Top 15 Oficial de Convicción Logit (Horizonte 30 Días)")
        st.write("Selección de mayor asimetría matemática libre de trampas de valor:")

        top15 = df_aprobadas.head(15).copy()
        top15['Ranking'] = range(1, len(top15) + 1)

        columnas_top15 = [
            'Ranking', 'Ticker', 'Nombre', 'Sector', 'Clasificacion',
            'Probabilidad_Logit', 'Precio_Actual', 'Target_WallSt', 
            'Upside_B5_%', 'PEG_Ratio', 'Margen_Op_%', 'Crec_Ventas_%'
        ]

        st.dataframe(
            top15[columnas_top15].style.format({
                'Probabilidad_Logit': '{:.2%}',
                'Precio_Actual': '${:.2f}',
                'Target_WallSt': '${:.2f}',
                'Upside_B5_%': '+{:.2f}%',
                'PEG_Ratio': '{:.2f}x',
                'Margen_Op_%': '{:.1f}%',
                'Crec_Ventas_%': '{:.1f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # RESUMEN EJECUTIVO
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Candidatas Aprobadas", len(df_aprobadas))
        with c2: st.metric("Descartadas por Filtro", len(df_vetadas))
        with c3: st.metric("Gangas (>85% Prob)", len(df_aprobadas[df_aprobadas['Probabilidad_Logit'] >= 0.85]))
        with c4: st.metric("Margen Op Promedio Top 15", f"{top15['Margen_Op_%'].mean():.1f}%")

        # PESTAÑAS DE INSPECCIÓN
        tab_full, tab_vetadas = st.tabs(["📋 Universo Completo Filtrado", "🛡️ Bitácora de Exclusión (Vetadas)"])

        with tab_full:
            cols_full = [
                'Ticker', 'Nombre', 'Sector', 'Clasificacion', 'Probabilidad_Logit',
                'Precio_Actual', 'Target_WallSt', 'Upside_B5_%', 'PEG_Ratio', 
                'Margen_Op_%', 'Crec_Ventas_%', 'Dif_%_vs_Max'
            ]
            st.dataframe(
                df_aprobadas[cols_full].style.format({
                    'Probabilidad_Logit': '{:.2%}',
                    'Precio_Actual': '${:.2f}',
                    'Target_WallSt': '${:.2f}',
                    'Upside_B5_%': '+{:.2f}%',
                    'PEG_Ratio': '{:.2f}x',
                    'Margen_Op_%': '{:.1f}%',
                    'Crec_Ventas_%': '{:.1f}%',
                    'Dif_%_vs_Max': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

        with tab_vetadas:
            st.write("Emisoras del tablero descartadas automáticamente:")
            st.dataframe(df_vetadas, use_container_width=True, hide_index=True)
    else:
        st.warning("No se encontraron activos que cumplan con los filtros de calidad en el archivo actual.")
else:
    st.info("Coloca un archivo con el formato `*export.csv` en la raíz de la aplicación para procesar el modelo.")
