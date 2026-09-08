import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import glob
import time
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="SBS Quant Lab - Logit Autónomo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
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
# UNIVERSO Y PARÁMETROS DEL TABLERO MÁXIMO
# ==============================================================================
UNIVERSO_TICKERS = [
    "ADYEN.AS", "UBER", "ADP", "DSY.PA", "UNH", "TEM", "OSCR", "HIMS", "DECK", "ADBE", 
    "ACN", "DLO", "FDS", "WKL.AS", "LULU", "NVO", "GEV", "BE", "VRT", "CEG", 
    "NEE", "SRE", "VST", "V", "MA", "MCO", "SPGI", "ISRG", "AXON", "ABNB", 
    "ANET", "BSX", "TTD", "NOW", "CRM", "SCHW", "BLK", "GS", "XOM", "CVX", 
    "CAT", "DE", "FIX", "ETN", "HON", "WM", "SMCI", "ALAB", "CORT", "ONTO", 
    "AX", "VIV", "GHM", "SLAB", "LSCC", "LASR", "SITM", "MCHP", "MRVL", "BAM", 
    "DHR", "QSR", "BABA", "GE", "CPNG", "EXPE", "ROK", "ZBRA", "CGNX", "PATH", 
    "PEGA", "MDT", "PRCT", "OMCL", "SYK", "TER", "LECO", "OII", "FARO", "PTC", 
    "QCOM", "AVAV", "TDY", "KTOS", "NOC", "GD", "RTX", "LHX", "APP", "IREN", 
    "AMAT", "KLAC", "RMBS", "SIMO", "ARM", "SNPS", "CRDO", "GLW", "AMKR", "PWR", 
    "CCJ", "BWXT", "UUUU", "TMQ", "UAMY", "MP", "FCX", "TECK", "SCCO", "IONQ", 
    "RGTI", "COIN", "SPOT", "DDOG", "RXRX", "POET", "RBLX", "CRCL", "BMNR", "ACHR", 
    "BEAM", "MOH", "ENB", "TOST", "AMGN", "FOX", "UTHR", "GOLD", "WBA", "JNJ", 
    "HD", "ABBV", "O", "BLDR", "TPL", "FICO", "DPZ", "URI", "BKNG", "MNST", 
    "WDAY", "SOFI", "NU", "NVDA", "AMD", "TSM", "AVGO", "MU", "ASML", "LRCX", 
    "PANW", "CRWD", "FTNT", "ZS", "OKTA", "SNOW", "PLTR", "LLY", "VRTX", "REGN", 
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "MARA", "RIOT", "WMT", "TGT", "COST", 
    "NFLX", "TSLA", "PYPL", "SHOP", "SE", "IBM", "QBTS", "ONDS", "MVST", "ASTS", 
    "NBIS", "RKLB", "FSLR", "EC", "PL", "BA", "SATS", "IRDM", "RDW", "LIN", 
    "GFS", "COHR", "LITE", "INTC", "GENB", "OUST", "PRME", "RVMD", "NXP", "TXN", 
    "ADI", "LAM", "CBRS", "OSS", "PENG", "STRL", "ZETA", "CSCO", "AXTI", "SNDK", 
    "RDDT", "LUNR", "LLAP", "VIAV", "AEVA", "SPIR", "ARQQ", "LAZR", "MTSI", "GILT", 
    "SALT", "TWST", "CLSK", "LEU", "SMR", "ZTS", "AAOI", "OKLO", "VPG", "SYM", 
    "INFQ", "USAR", "ROKU", "CRSP", "INSM", "UI", "APLD", "VSAT", "PGY", "BETR", 
    "TMC", "LTBR", "GRAL", "OPEN", "CIFR", "NVTS"
]

EXCLUIR_TICKERS = [
    'BTC-USD', 'ETH-USD', 'CONL', 'SOXL', 'LABU', 'MARA', 'CLSK', 'CIFR', 
    'IREN', 'RGTI', 'QBTS', 'POET', 'OPEN', 'BETR', 'MVST', 'UAMY', 'TMQ', 
    'SMR', 'LTBR'
]

# Pesos del Modelo Logit
BETA_0 =  0.20
BETA_1 = -1.50   # Factor PEG (Menor es mejor)
BETA_2 =  1.20   # Factor Margen Operativo % (Mayor es mejor)
BETA_3 =  1.10   # Factor Crecimiento Ventas % (Mayor es mejor)
BETA_4 = -0.90   # Factor Dif % vs Max (Mayor caída suma)
BETA_5 =  1.30   # Factor Upside Wall Street % (Mayor es mejor)

CSV_MAESTRO = "tablero_export_auto.csv"

# ==============================================================================
# MOTOR DE DESCARGA AUTÓNOMO (CREA EL EXPORT SI NO EXISTE O ESTÁ VIEJO)
# ==============================================================================
def construir_tablero_maximo_autonomo():
    """Descarga datos frescos de yfinance y construye el DataFrame idéntico al Tablero Máximo."""
    registros = []
    
    # 1. Ingesta masiva de precios históricos (1 año para calcular el máximo de 52 semanas)
    try:
        data_hist = yf.download(UNIVERSO_TICKERS, period="1y", interval="1d", group_by='ticker', progress=False)
    except Exception:
        data_hist = None

    for sym in UNIVERSO_TICKERS:
        try:
            tk = yf.Ticker(sym)
            inf = tk.info
            
            # Obtener serie de precios
            if data_hist is not None and sym in data_hist.columns.levels[0]:
                df_sym = data_hist[sym].dropna()
            else:
                df_sym = tk.history(period="1y")

            if df_sym.empty:
                continue

            precio_act = float(df_sym['Close'].iloc[-1])
            max_52w = float(df_sym['High'].max())
            dif_vs_max = ((precio_act - max_52w) / max_52w) * 100.0 if max_52w > 0 else 0.0

            # Extracción de variables financieras
            nombre = inf.get("shortName", sym)
            sector = inf.get("sector", "Tecnología")
            margen_op = (inf.get("operatingMargins", 0.0) or 0.0) * 100.0
            crec_ventas = (inf.get("revenueGrowth", 0.0) or 0.0) * 100.0
            crec_eps = (inf.get("earningsGrowth", 0.0) or 0.0) * 100.0
            pe_act = inf.get("forwardPE", inf.get("trailingPE", 0.0)) or 0.0
            peg = inf.get("pegRatio", np.nan)
            target_ws = inf.get("targetMeanPrice", precio_act) or precio_act
            upside_b5 = ((target_ws - precio_act) / precio_act) * 100.0 if precio_act > 0 else 0.0

            registros.append({
                "Ticker": sym,
                "Nombre": nombre,
                "Sector": sector,
                "Precio_Actual": round(precio_act, 2),
                "Dif_%_vs_Max": round(dif_vs_max, 2),
                "PE_Actual": round(pe_act, 2),
                "PEG_Ratio": round(peg, 2) if pd.notna(peg) else np.nan,
                "Margen_Op_%": round(margen_op, 2),
                "Crec_EPS_%": round(crec_eps, 2),
                "Crec_Ventas_%": round(crec_ventas, 2),
                "Target_WallSt": round(target_ws, 2),
                "Upside_B5_%": round(upside_b5, 2)
            })
            time.sleep(0.01)
        except Exception:
            continue

    df_export = pd.DataFrame(registros)
    if not df_export.empty:
        df_export.to_csv(CSV_MAESTRO, index=False)
    return df_export

# ==============================================================================
# MOTOR MATEMÁTICO LOGIT
# ==============================================================================
def calcular_modelo_logit(df_raw):
    """Aplica los filtros de veto y calcula el ranking Logit a 30 días."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    vetadas, aprobadas_idx = [], []

    for idx, row in df_raw.iterrows():
        t = str(row.get('Ticker', ''))
        if t in EXCLUIR_TICKERS:
            vetadas.append({"Ticker": t, "Nombre": row.get('Nombre', ''), "Razón": "Cripto/Apalancado/Volátil"})
            continue
        if row.get('Precio_Actual', 0) < 15.0:
            vetadas.append({"Ticker": t, "Nombre": row.get('Nombre', ''), "Razón": "Precio < $15 USD"})
            continue
        if row.get('Margen_Op_%', 0) < 12.0:
            vetadas.append({"Ticker": t, "Nombre": row.get('Nombre', ''), "Razón": f"Margen Op. bajo ({row.get('Margen_Op_%', 0):.1f}%)"})
            continue
        if row.get('Crec_Ventas_%', 0) < 5.0:
            vetadas.append({"Ticker": t, "Nombre": row.get('Nombre', ''), "Razón": f"Ventas estancadas ({row.get('Crec_Ventas_%', 0):.1f}%)"})
            continue
        if row.get('Crec_EPS_%', 0) < 0.0:
            vetadas.append({"Ticker": t, "Nombre": row.get('Nombre', ''), "Razón": f"Contracción en utilidades ({row.get('Crec_EPS_%', 0):.1f}%)"})
            continue
        if row.get('Upside_B5_%', 0) < 5.0:
            vetadas.append({"Ticker": t, "Nombre": row.get('Nombre', ''), "Razón": f"Upside consenso < 5% ({row.get('Upside_B5_%', 0):.1f}%)"})
            continue
        if pd.isna(row.get('PEG_Ratio')) or row.get('PEG_Ratio', 0) <= 0:
            vetadas.append({"Ticker": t, "Nombre": row.get('Nombre', ''), "Razón": "PEG distorsionado"})
            continue

        aprobadas_idx.append(idx)

    df_base = df_raw.loc[aprobadas_idx].copy()
    if df_base.empty:
        return pd.DataFrame(), pd.DataFrame(vetadas)

    cols = ['PEG_Ratio', 'Margen_Op_%', 'Crec_Ventas_%', 'Dif_%_vs_Max', 'Upside_B5_%']
    for c in cols:
        mean_val = df_base[c].mean()
        std_val = df_base[c].std() if df_base[c].std() > 0 else 1.0
        df_base[f'z_{c}'] = (df_base[c] - mean_val) / std_val

    df_base['score_z'] = (
        BETA_0 +
        BETA_1 * df_base['z_PEG_Ratio'] +
        BETA_2 * df_base['z_Margen_Op_%'] +
        BETA_3 * df_base['z_Crec_Ventas_%'] +
        BETA_4 * df_base['z_Dif_%_vs_Max'] +
        BETA_5 * df_base['z_Upside_B5_%']
    )
    df_base['Probabilidad_Logit'] = 1.0 / (1.0 + np.exp(-df_base['score_z']))

    conds = [
        (df_base['Probabilidad_Logit'] >= 0.85),
        (df_base['Probabilidad_Logit'] >= 0.70),
        (df_base['Probabilidad_Logit'] >= 0.55),
        (df_base['Probabilidad_Logit'] >= 0.40)
    ]
    labels = ["💎 GANGA INSTITUCIONAL", "🟢 Muy Fuerte", "🟡 Atractiva", "⚪ Neutral"]
    df_base['Clasificacion'] = np.select(conds, labels, default="🔴 Descartar")

    df_aprobadas = df_base.sort_values(by=['Probabilidad_Logit', 'Margen_Op_%'], ascending=[False, False]).reset_index(drop=True)
    df_vetadas = pd.DataFrame(vetadas)

    df_aprobadas.to_csv("logit_data.csv", index=False)
    df_vetadas.to_csv("vetados_quality.csv", index=False)
    return df_aprobadas, df_vetadas

# ==============================================================================
# GESTOR DE DATOS EN MEMORIA / CACHÉ
# ==============================================================================
@st.cache_data(ttl=14400) # Se recalcula automáticamente cada 4 horas
def obtener_datos_actualizados():
    """Verifica la antigüedad del archivo local. Si no existe o tiene >4 horas, lo genera."""
    necesita_descarga = True
    
    # 1. Buscar cualquier export previo en el directorio
    archivos_existentes = glob.glob("*export*.csv") + glob.glob("tablero*.csv")
    if archivos_existentes:
        archivos_existentes.sort(key=os.path.getmtime, reverse=True)
        mas_reciente = archivos_existentes[0]
        edad_horas = (time.time() - os.path.getmtime(mas_reciente)) / 3600.0
        
        # Si tiene menos de 4 horas, usarlo sin hacer esperar al usuario
        if edad_horas < 4.0:
            df_raw = pd.read_csv(mas_reciente)
            necesita_descarga = False
    
    if necesita_descarga:
        df_raw = construir_tablero_maximo_autonomo()

    return calcular_modelo_logit(df_raw)

# ==============================================================================
# INTERFAZ STREAMLIT
# ==============================================================================
st.title("🧠 SBS Center - Laboratorio Logit 100% Autónomo")
st.caption("Auto-actualización en tiempo real | Top 15 de Convicción Fundamental a 30 Días")
st.markdown("---")

with st.sidebar:
    st.header("🔄 Estado del Sistema")
    if os.path.exists(CSV_MAESTRO):
        f_time = datetime.fromtimestamp(os.path.getmtime(CSV_MAESTRO)).strftime('%Y-%m-%d %H:%M')
        st.success(f"Base de datos activa.\nÚltimo cálculo: {f_time}")
    else:
        st.info("Inicializando datos por primera vez...")

    if st.button("⚡ Forzar Actualización Completa Ahora", use_container_width=True):
        with st.spinner("Descargando mercado completo en vivo... (tomará ~30s)"):
            st.cache_data.clear()
            df_aprobadas, df_vetadas = obtener_datos_actualizados()
            st.success("¡Base de datos y modelo actualizados!")
            st.rerun()

    st.divider()
    st.markdown("""
    **Reglas de Ejecución:**
    * Sin archivos manuales requeridos.
    * Auto-expiración de caché: **4 Horas**.
    * Salida anticipada recomendada: **+8% a +10%**.
    """)

# Carga y renderizado automático
df_aprobadas, df_vetadas = obtener_datos_actualizados()

if df_aprobadas is not None and not df_aprobadas.empty:
    st.markdown("<div class='top15-box'>", unsafe_allow_html=True)
    st.subheader("🔥 Top 15 Oficial de Convicción Logit (Horizonte 30 Días)")
    st.write("Generado automáticamente evaluando calidad operativa, PEG y sobreventa:")

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

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Empresas Calificadas", len(df_aprobadas))
    with c2: st.metric("Empresas Vetadas (Filtro Sanitario)", len(df_vetadas))
    with c3: st.metric("Margen Op. Promedio Top 15", f"{top15['Margen_Op_%'].mean():.1f}%")

    tab1, tab2 = st.tabs(["📋 Universo Completo Calificado", "🛡️ Bitácora de Empresas Vetadas"])
    with tab1:
        cols_all = ['Ticker', 'Nombre', 'Sector', 'Clasificacion', 'Probabilidad_Logit', 'Precio_Actual', 'Target_WallSt', 'Upside_B5_%', 'PEG_Ratio', 'Margen_Op_%', 'Crec_Ventas_%', 'Dif_%_vs_Max']
        st.dataframe(df_aprobadas[cols_all].style.format({
            'Probabilidad_Logit': '{:.2%}', 'Precio_Actual': '${:.2f}', 'Target_WallSt': '${:.2f}',
            'Upside_B5_%': '+{:.2f}%', 'PEG_Ratio': '{:.2f}x', 'Margen_Op_%': '{:.1f}%',
            'Crec_Ventas_%': '{:.1f}%', 'Dif_%_vs_Max': '{:.1f}%'
        }), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(df_vetadas, use_container_width=True, hide_index=True)
