import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import random
import os
from datetime import datetime

# ==========================================================================
# 1. MOTOR MAESTRO DE INGESTA Y CÁLCULO CUANTITATIVO (EXTRACTOR LOGIT)
# ==========================================================================
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

UMBRAL_ROIC = 0.10          
UMBRAL_COBERTURA = 4.5      
UMBRAL_MARGEN_BRUTO_ESTANDAR = 0.35  
UMBRAL_MARGEN_BRUTO_SEMIS = 0.15
SECTOR_HARDWARE_SEMIS = ["NVDA", "AMD", "TSM", "AVGO", "MU", "ASML", "LRCX", "AMAT", "KLAC", "INTC", "TXN", "ADI", "LAM", "SMCI", "QCOM"]

def ejecutar_pipeline_cuantitativo():
    """Descarga, filtra y procesa los 150 tickers bajo la matriz Logit en menos de 30 segundos."""
    aprobadas, vetadas = [], []
    fecha_hoy_str = datetime.today().strftime('%Y-%m-%d')
    
    # Descarga vectorizada de alta velocidad
    try:
        precios_bloque = yf.download(UNIVERSO_TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
    except:
        precios_bloque = None

    for ticker in UNIVERSO_TICKERS:
        time.sleep(0.05)  # Micro-delay preventivo
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            if precios_bloque is not None and ticker in precios_bloque.columns.levels[0]:
                history_2y = precios_bloque[ticker].dropna()
            else:
                history_2y = t.history(period="2y")
                
            if history_2y.empty:
                vetadas.append({"Ticker": ticker, "Razón": "Falta de historial en API"})
                continue
                
            margen_bruto = info.get("grossMargins", 0.0) or 0.0
            margen_neto = info.get("profitMargins", 0.0) or 0.0
            roe = info.get("returnOnEquity", 0.0) or 0.0
            ebit = info.get("operatingMargins", 0.0) * info.get("totalRevenue", 1.0) if info.get("operatingMargins") else 0.0
            roic = info.get("returnOnAssets", 0.0) or roe or 0.0
            cobertura_interes = 999.0
            
            req_margen_bruto = UMBRAL_MARGEN_BRUTO_SEMIS if ticker in SECTOR_HARDWARE_SEMIS else UMBRAL_MARGEN_BRUTO_ESTANDAR
            
            if margen_bruto < req_margen_bruto:
                vetadas.append({"Ticker": ticker, "Razón": f"Margen Bruto insuficiente ({margen_bruto*100:.1f}%)"})
                continue
            if margen_neto <= 0:
                vetadas.append({"Ticker": ticker, "Razón": f"Margen Neto no positivo ({margen_neto*100:.1f}%)"})
                continue
                
            pe_actual = info.get("trailingPE", info.get("forwardPE", 0.0)) or 0.0
            fcf_yield = (info.get("freeCashflow", 0.0) / info.get("marketCap", 1.0)) if info.get("marketCap", 1.0) > 0 else 0.0
            
            close_prices = history_2y['Close'].resample('ME').last()
            pe_series = [pe_actual * (p / close_prices.iloc[-1]) for p in close_prices]
            
            p20_pe = np.percentile(pe_series, 20) if pe_series else 0.0
            p80_pe = np.percentile(pe_series, 80) if pe_series else 1.0
            mean_pe = np.mean(pe_series) if pe_series else 0.0
            std_pe = np.std(pe_series) if pe_series and np.std(pe_series) > 0 else 1.0
            
            x_pe_percentil = (pe_actual - p20_pe) / (p80_pe - p20_pe) if (p80_pe - p20_pe) > 0 else 0.5
            x_pe_percentil = max(0.0, min(1.0, x_pe_percentil))
            z_pe = (pe_actual - mean_pe) / std_pe
            
            score_z = 1.8 - 3.5 * x_pe_percentil - 1.2 * z_pe + 2.0 * fcf_yield
            probabilidad = 1 / (1 + np.exp(-score_z))
            
            if probabilidad >= 0.85: categoria = "💎 GANGA"
            elif probabilidad >= 0.71: categoria = "Muy Barata"
            elif probabilidad >= 0.56: categoria = "Barata"
            elif probabilidad >= 0.45: categoria = "Media"
            elif probabilidad >= 0.30: categoria = "Cara"
            elif probabilidad >= 0.15: categoria = "Muy Cara"
            else: categoria = "🚫 EVITAR"
            
            aprobadas.append({
                "Ticker": ticker, "PE_Actual": pe_actual, "Percentil_PE_24M": x_pe_percentil,
                "Z_Score_PE": z_pe, "FCF_Yield": fcf_yield, "Margen_Bruto": margen_bruto,
                "Delta_Margen_Bruto": 0.0, "ROIC": roic, "Cobertura_Interes": cobertura_interes,
                "Probabilidad_Logit": probabilidad, "Clasificacion": categoria,
                "Ultima_Actualizacion": fecha_hoy_str
            })
        except:
            continue

    if aprobadas: pd.DataFrame(aprobadas).to_csv("logit_data.csv", index=False)
    if vetadas: pd.DataFrame(vetadas).to_csv("vetados_quality.csv", index=False)

# ==========================================================================
# 2. INTERFAZ GRÁFICA PREMIUM EN STREAMLIT
# ==========================================================================
st.set_page_config(page_title="SBS Quant Lab - Logit Unificado", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card { background-color: #f8fafc; padding: 18px; border-radius: 6px; border-left: 5px solid #1e3a8a; margin-bottom: 12px; }
    .top10-container { background-color: #fffbf5; padding: 22px; border-radius: 8px; border: 1px solid #fed7aa; border-left: 6px solid #ea580c; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SBS Center - Laboratorio Analítico de Valuación Logit")
st.caption("Ecosistema Cuantitativo 'The One Ring' | Suite de Ejecución Monolítica Total")
st.markdown("---")

def get_file_timestamp(filepath):
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0

@st.cache_data(ttl=600)
def cargar_datos(ts_clean, ts_vetadas):
    df_clean = pd.read_csv("logit_data.csv") if os.path.exists("logit_data.csv") else None
    df_vetadas = pd.read_csv("vetados_quality.csv") if os.path.exists("vetados_quality.csv") else None
    return df_clean, df_vetadas

ts_clean = get_file_timestamp("logit_data.csv")
ts_vetadas = get_file_timestamp("vetados_quality.csv")
df_clean, df_vetadas = cargar_datos(ts_clean, ts_vetadas)

# BARRA LATERAL CON PROCESAMIENTO INTERNO EN VIVO
with st.sidebar:
    st.header("🔄 Control de Ingesta")
    st.write(f"Último cálculo físico: {datetime.fromtimestamp(ts_clean).strftime('%Y-%m-%d %H:%M') if ts_clean > 0 else 'Ninguno'}")
    
    if st.button("🚀 Ejecutar Algoritmo en Tiempo Real", use_container_width=True):
        with st.spinner("Triturando estados financieros y cotizaciones de apertura..."):
            ejecutar_pipeline_cuantitativo()
            st.success("¡Matriz recalculada exitosamente!")
            st.cache_data.clear()
            st.rerun()
            
    st.divider()
    st.header("📊 Filtros")
    categoria_sel = st.selectbox("Filtrar Universo General por Convicción:", ["Todos"] + list(sorted(df_clean["Clasificacion"].unique()))) if df_clean is not None else "Todos"

# DESPLIEGUE GENERAL DE LA SALA DE TRADING
if df_clean is not None:
    st.markdown("<div class='top10-container'>", unsafe_allow_html=True)
    st.subheader("🔥 El Top 10 de Convicción Absoluta SBS (Selección de Capital Eficiente)")
    
    df_gangas = df_clean[df_clean["Clasificacion"].isin(["💎 GANGA", "Muy Barata"])].copy()
    if not df_gangas.empty:
        df_top10 = df_gangas.sort_values(by=["Probabilidad_Logit", "FCF_Yield"], ascending=[False, False]).head(10)
        
        def color_gangas(row):
            return ['background-color: #f0fff4; color: #166534; font-weight: bold;' if row["Clasificacion"] == "💎 GANGA" else '' for _ in row]

        styled_top10 = (df_top10[["Ticker", "Clasificacion", "Probabilidad_Logit", "PE_Actual", "Percentil_PE_24M", "FCF_Yield", "ROIC"]]
                        .style.apply(color_gangas, axis=1)
                        .format({"Probabilidad_Logit": lambda x: f"{x*100:.2f}%", "FCF_Yield": lambda x: f"{x*100:.2f}%", "Percentil_PE_24M": lambda x: f"{x*100:.1f}%", "ROIC": lambda x: f"{x*100:.1f}%", "PE_Actual": "{:.2f}v"}))
        st.dataframe(styled_top10, use_container_width=True, hide_index=True)
    else:
        st.info("🎯 El mercado cotiza en valuaciones elevadas. No hay activos en zona de estrés que activen el Top 10 hoy.")
    st.markdown("</div>", unsafe_allow_html=True)

    # MÉTRICAS GLOBALES
    n_clean, n_vetadas = len(df_clean), (len(df_vetadas) if df_vetadas is not None else 0)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: st.metric("Universo Analizado", n_clean + n_vetadas)
    with col_m2: st.metric("Aprobadas por Calidad", n_clean, delta=f"{(n_clean/(n_clean+n_vetadas)*100):.1f}%")
    with col_m3: st.metric("GANGAS en Radar", len(df_clean[df_clean["Clasificacion"] == "💎 GANGA"]))
    with col_m4: st.metric("Vetadas por Riesgo", n_vetadas)

    tab_modelo, tab_buscador, tab_auditoria = st.tabs(["📈 Universo Completo y Señales", "🔍 Reporte por Ticker", "🛡️ Auditoría de Exclusión"])

    with tab_modelo:
        st.bar_chart(df_clean["Clasificacion"].value_counts())
        df_filtrado = df_clean if categoria_sel == "Todos" else df_clean[df_clean["Clasificacion"] == categoria_sel]
        st.dataframe(df_filtrado[["Ticker", "Clasificacion", "Probabilidad_Logit", "PE_Actual", "Percentil_PE_24M", "Z_Score_PE", "FCF_Yield", "Margen_Bruto", "ROIC"]].style.format({"Probabilidad_Logit": lambda x: f"{x*100:.2f}%", "Percentil_PE_24M": lambda x: f"{x*100:.1f}%", "FCF_Yield": lambda x: f"{x*100:.2f}%", "Margen_Bruto": lambda x: f"{x*100:.1f}%", "ROIC": lambda x: f"{x*100:.1f}%", "PE_Actual": "{:.2f}v", "Z_Score_PE": "{:.2f}"}), use_container_width=True, hide_index=True)

    with tab_buscador:
        ticker_buscar = st.selectbox("Selecciona un Ticker:", sorted(df_clean["Ticker"].unique()))
        if ticker_buscar:
            row = df_clean[df_clean["Ticker"] == ticker_buscar].iloc[0]
            st.markdown(f"### Análisis de Vectores para **{ticker_buscar}**")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1: st.markdown(f"<div class='metric-card'><h4>Clasificación</h4><h2>{row['Clasificacion']}</h2></div>", unsafe_allow_html=True)
            with col_b2: st.markdown(f"<div class='metric-card'><h4>Probabilidad Upside</h4><h2>{row['Probabilidad_Logit']*100:.2f}%</h2></div>", unsafe_allow_html=True)
            with col_b3: st.markdown(f"<div class='metric-card'><h4>P/E Coetáneo</h4><h2>{row['PE_Actual']:.2f}v</h2></div>", unsafe_allow_html=True)

    with tab_auditoria:
        if df_vetadas is not None and not df_vetadas.empty:
            razon_sel = st.selectbox("Filtrar Rechazo:", ["Todas"] + list(df_vetadas["Razón"].unique()))
            st.dataframe(df_vetadas if razon_sel == "Todas" else df_vetadas[df_vetadas["Razón"] == razon_sel], use_container_width=True, hide_index=True)
else:
    st.warning("⚠️ Sin base de datos local. Presiona el botón de la barra lateral para calcular el modelo por primera vez.")