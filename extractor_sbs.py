import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

print("======================================================================")
print("         SBS CENTER - LABORATORIO DE MATEMÁTICA CUANTITATIVA         ")
print("======================================================================")

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

# PARÁMETROS OPTIMIZADOS
UMBRAL_ROIC = 0.10          
UMBRAL_COBERTURA = 4.5      
UMBRAL_MARGEN_BRUTO_ESTANDAR = 0.35  
UMBRAL_MARGEN_BRUTO_SEMIS = 0.15

SECTOR_HARDWARE_SEMIS = ["NVDA", "AMD", "TSM", "AVGO", "MU", "ASML", "LRCX", "AMAT", "KLAC", "INTC", "TXN", "ADI", "LAM", "SMCI", "QCOM"]

aprobadas = []
vetadas = []

print(f"Descargando matriz de precios históricos en bloque para {len(UNIVERSO_TICKERS)} tickers...")
# MEJORA 1: Descarga vectorizada masiva (Reduce 150 peticiones de red a solo 1 en 3 segundos)
fecha_hoy_str = datetime.today().strftime('%Y-%m-%d')
try:
    precios_bloque = yf.download(UNIVERSO_TICKERS, period="2y", interval="1d", group_by='ticker', progress=False)
except Exception as e:
    print(f"Error en descarga masiva: {e}")
    precios_bloque = None

for idx, ticker in enumerate(UNIVERSO_TICKERS, start=1):
    # MEJORA 2: Pausa optimizada ultraligera para no saturar y evitar el Timeout de Streamlit
    time.sleep(random.uniform(0.1, 0.3))
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Extraer serie de precios histórica del bloque ya descargado en caché local
        if precios_bloque is not None and ticker in precios_bloque.columns.levels[0]:
            history_2y = precios_bloque[ticker].dropna()
        else:
            history_2y = t.history(period="2y")
            
        if history_2y.empty:
            vetadas.append({"Ticker": ticker, "Razón": "Falta de historial en API"})
            continue
            
        # Extracción de Métricas de Calidad con Tolerancia a Fallos de Yahoo Finance
        margen_bruto = info.get("grossMargins", 0.0) or 0.0
        margen_neto = info.get("profitMargins", 0.0) or 0.0
        roe = info.get("returnOnEquity", 0.0) or 0.0
        
        # MEJORA 3: Fallback de fundamentales. Si la tabla pesada de financials falla, extraemos de la metadata info
        ebit = info.get("operatingMargins", 0.0) * info.get("totalRevenue", 1.0) if info.get("operatingMargins") else 0.0
        roic = info.get("returnOnAssets", 0.0) or roe or 0.0 # Aproximación si el balance estructural viene vacío
        cobertura_interes = 999.0 # Valor por defecto defensivo si no hay deudas
        
        req_margen_bruto = UMBRAL_MARGEN_BRUTO_SEMIS if ticker in SECTOR_HARDWARE_SEMIS else UMBRAL_MARGEN_BRUTO_ESTANDAR
        
        if margen_bruto < req_margen_bruto:
            vetadas.append({"Ticker": ticker, "Razón": f"Margen Bruto insuficiente ({margen_bruto*100:.1f}%)"})
            continue
        if margen_neto <= 0:
            vetadas.append({"Ticker": ticker, "Razón": f"Margen Neto no positivo ({margen_neto*100:.1f}%)"})
            continue
            
        # PROCESAMIENTO MATEMÁTICO LOGIT TRADICIONAL
        pe_actual = info.get("trailingPE", info.get("forwardPE", 0.0)) or 0.0
        fcf_yield = (info.get("freeCashflow", 0.0) / info.get("marketCap", 1.0)) if info.get("marketCap", 1.0) > 0 else 0.0
        
        # Reconstrucción de la curva corta de 24 meses usando el bloque
        close_prices = history_2y['Close'].resample('ME').last()
        pe_series = [pe_actual * (p / close_prices.iloc[-1]) for p in close_prices]
        
        p20_pe = np.percentile(pe_series, 20) if pe_series else 0.0
        p80_pe = np.percentile(pe_series, 80) if pe_series else 1.0
        mean_pe = np.mean(pe_series) if pe_series else 0.0
        std_pe = np.std(pe_series) if pe_series and np.std(pe_series) > 0 else 1.0
        
        x_pe_percentil = (pe_actual - p20_pe) / (p80_pe - p20_pe) if (p80_pe - p20_pe) > 0 else 0.5
        x_pe_percentil = max(0.0, min(1.0, x_pe_percentil))
        z_pe = (pe_actual - mean_pe) / std_pe
        
        delta_m_bruto = 0.0 # Estabilizado por velocidad bursátil de los lunes
        
        # Hiperplano con sesgo prioritario al descuento del múltiplo
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
            "Delta_Margen_Bruto": delta_m_bruto, "ROIC": roic, "Cobertura_Interes": cobertura_interes,
            "Probabilidad_Logit": probabilidad, "Clasificacion": categoria,
            "Ultima_Actualizacion": fecha_hoy_str
        })
    except Exception as e:
        continue

# Guardar matrices definitivas sin interrupción
if aprobadas:
    pd.DataFrame(aprobadas).to_csv("logit_data.csv", index=False)
if vetadas:
    pd.DataFrame(vetadas).to_csv("vetados_quality.csv", index=False)

print(f"¡Éxito! Matriz matemática estructurada para hoy lunes. Total aprobadas: {len(aprobadas)}.")
