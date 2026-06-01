import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import time

# ==========================================================================
# 1. CONFIGURACIÓN CORPORATIVA E INSTITUCIONAL (SALA DE TRADING)
# ==========================================================================
st.set_page_config(
    page_title="SBS Quant Lab - Motor Logit v2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS premium unificados para la sala de trading de Isengard
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8fafc;
        padding: 18px;
        border-radius: 6px;
        border-left: 5px solid #1e3a8a;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .top10-container {
        background-color: #fffbf5;
        padding: 22px;
        border-radius: 8px;
        border: 1px solid #fed7aa;
        border-left: 6px solid #ea580c;
        margin-bottom: 25px;
    }
    .ticker-badge {
        background-color: #1e3a8a;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SBS Center - Laboratorio Analítico de Valuación Logit")
st.caption("Ecosistema Cuantitativo 'The One Ring' | Módulo 4: Análisis de Múltiplos por Regresión Estructural")
st.markdown("---")

# ==========================================================================
# 2. SISTEMA DE CONTROL DE CACHÉ DINÁMICO POR MUTACIÓN DE ARCHIVO
# ==========================================================================
def get_file_timestamp(filepath):
    """Devuelve el timestamp de la última modificación para romper el caché automáticamente."""
    if os.path.exists(filepath):
        return os.path.getmtime(filepath)
    return 0.0

# El caché se invalida por completo si el archivo físico en el servidor muta o cambia de tamaño
@st.cache_data(ttl=600)
def cargar_datos(ts_clean, ts_vetadas):
    try:
        df_clean = pd.read_csv("logit_data.csv") if os.path.exists("logit_data.csv") else None
        df_vetadas = pd.read_csv("vetados_quality.csv") if os.path.exists("vetados_quality.csv") else None
        return df_clean, df_vetadas
    except Exception as e:
        st.error(f"Error de lectura actuarial en bases de datos: {e}")
        return None, None

# Obtener marcas de tiempo vivas antes de llamar al cargador
ts_clean = get_file_timestamp("logit_data.csv")
ts_vetadas = get_file_timestamp("vetados_quality.csv")
df_clean, df_vetadas = cargar_datos(ts_clean, ts_vetadas)

# ==========================================================================
# 3. BARRA LATERAL: PANEL DE SINCRONIZACIÓN Y FILTROS INTEGRADOS
# ==========================================================================
with st.sidebar:
    st.header("🔄 Control Operativo")
    st.info("Módulo de sincronización remota de Isengard. Utiliza este control si los datos de los lunes por la mañana no se actualizan en la apertura.")
    
    # INTERRUPTOR MAESTRO: Manda llamar al extractor optimizado directamente en el servidor cloud
    if st.button("🚀 Forzar Ingesta y Recálculo Logit", use_container_width=True):
        with st.spinner("Triturando vectores fundamentales en extractor_sbs.py..."):
            try:
                # Ejecución nativa del proceso hermano de alta velocidad
                resultado = subprocess.run(["python", "extractor_sbs.py"], capture_output=True, text=True, timeout=120)
                if resultado.returncode == 0:
                    st.success("¡Base de datos unificada y actualizada con éxito!")
                    time.sleep(1)
                    st.cache_data.clear()  # Purga total de la memoria RAM del servidor
                    st.rerun()             # Refresco forzoso de la interfaz
                else:
                    st.error("El extractor falló en la compilación estructural.")
                    st.code(resultado.stderr)
            except subprocess.TimeoutExpired:
                st.error("Exceso de tiempo en la consulta remota de red.")
            except Exception as e:
                st.error(f"Error crítico de subproceso: {e}")
                
    st.divider()
    st.header("📊 Filtros de Exploración")
    if df_clean is not None:
        categorias_disponibles = ["Todos"] + list(sorted(df_clean["Clasificacion"].unique()))
        categoria_sel = st.selectbox("Filtrar Universo General por Convicción:", categorias_disponibles)
    else:
        categoria_sel = "Todos"

# ==========================================================================
# 4. DESPLIEGUE EXECUTIVO: SECCIÓN DE ALTA CONVICCIÓN SBS
# ==========================================================================
if df_clean is not None:
    
    st.markdown("<div class='top10-container'>", unsafe_allow_html=True)
    st.subheader("🔥 El Top 10 de Convicción Absoluta SBS (Selección de Capital Eficiente)")
    st.markdown("""
        *Reglas operativas de asignación de recursos:* Ejecutar compras preferentemente **lunes de 10:30 AM a 12:00 PM**. 
        Horizonte límite de salida: **90 días naturales** o en cuanto el activo recupere su valor medio (Clasificación 'Media'). 
        *No se utiliza Stop Loss comercial (el filtro de calidad operativo actúa como el único stop fundamental).*
    """)
    
    # Algoritmo de ordenamiento de doble capa: Probabilidad (Desc) -> FCF Yield (Desc)
    df_gangas = df_clean[df_clean["Clasificacion"].isin(["💎 GANGA", "Muy Barata"])].copy()
    
    if not df_gangas.empty:
        df_top10 = df_gangas.sort_values(
            by=["Probabilidad_Logit", "FCF_Yield"], 
            ascending=[False, False]
        ).head(10)
        
        # Formatear la tabla del Top 10 para despliegue ejecutivo
        df_top10_display = df_top10.copy()
        
        def color_gangas(row):
            return ['background-color: #f0fff4; color: #166534; font-weight: bold;' if row["Clasificacion"] == "💎 GANGA" else '' for _ in row]

        styled_top10 = (df_top10_display[["Ticker", "Clasificacion", "Probabilidad_Logit", "PE_Actual", "Percentil_PE_24M", "FCF_Yield", "ROIC"]]
                        .style.apply(color_gangas, axis=1)
                        .format({
                            "Probabilidad_Logit": lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x,
                            "FCF_Yield": lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x,
                            "Percentil_PE_24M": lambda x: f"{x*100:.1f}%" if isinstance(x, float) else x,
                            "ROIC": lambda x: f"{x*100:.1f}%" if isinstance(x, float) else x,
                            "PE_Actual": "{:.2f}v"
                        }))
        
        st.dataframe(styled_top10, use_container_width=True, height=380, hide_index=True)
    else:
        st.info("🎯 El mercado se encuentra cotizando en rangos de valuación justos o elevados. No hay activos en zona de estrés o descuento que activen el Top 10 actualmente.")
        
    st.markdown("</div>", unsafe_allow_html=True)

    # --- MÉTRICAS DE ESCALAFÓN GLOBAL ---
    n_clean = len(df_clean)
    n_vetadas = len(df_vetadas) if df_vetadas is not None else 0
    total_universo = n_clean + n_vetadas
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Universo Analizado", total_universo)
    with col_m2:
        pct_aprobadas = (n_clean / total_universo * 100) if total_universo > 0 else 0.0
        st.metric("Aprobadas por Calidad", n_clean, delta=f"{pct_aprobadas:.1f}%")
    with col_m3:
        gangas_count = len(df_clean[df_clean["Clasificacion"] == "💎 GANGA"])
        st.metric("GANGAS en el Radar", gangas_count, delta="Horizonte 90 Días", delta_color="inverse")
    with col_m4:
        st.metric("Vetadas por Riesgo", n_vetadas)

    # --- DISEÑO DE PESTAÑAS ANALÍTICAS ---
    tab_modelo, tab_buscador, tab_auditoria = st.tabs([
        "📈 Universo Completo y Señales", 
        "🔍 Reporte Actuarial por Ticker", 
        "🛡️ Auditoría de Exclusión (Vetadas)"
    ])

    # PESTAÑA 1: EXPLORACIÓN DEL MODELO GENERAL
    with tab_modelo:
        st.subheader("Distribución Estocástica del Universo")
        resumen_cat = df_clean["Clasificacion"].value_counts().reindex([
            "💎 GANGA", "Muy Barata", "Barata", "Media", "Cara", "Muy Cara", "🚫 EVITAR"
        ], fill_value=0)
        
        st.bar_chart(resumen_cat)
        
        st.subheader(f"Listado de Activos Filtrados: {categoria_sel}")
        df_filtrado = df_clean if categoria_sel == "Todos" else df_clean[df_clean["Clasificacion"] == categoria_sel]
        df_display = df_filtrado.copy()
        
        styled_general = (df_display[["Ticker", "Clasificacion", "Probabilidad_Logit", "PE_Actual", "Percentil_PE_24M", "Z_Score_PE", "FCF_Yield", "Margen_Bruto", "ROIC"]]
                          .style.format({
                              "Probabilidad_Logit": lambda x: f"{x*100:.2f}%",
                              "Percentil_PE_24M": lambda x: f"{x*100:.1f}%",
                              "FCF_Yield": lambda x: f"{x*100:.2f}%",
                              "Margen_Bruto": lambda x: f"{x*100:.1f}%",
                              "ROIC": lambda x: f"{x*100:.1f}%",
                              "PE_Actual": "{:.2f}v",
                              "Z_Score_PE": "{:.2f}"
                          }))
        st.dataframe(styled_general, use_container_width=True, height=500, hide_index=True)

    # PESTAÑA 2: BUSCADOR QUIRÚRGICO DE TICKERS (REPORTE DE AUDITORÍA)
    with tab_buscador:
        st.subheader("Análisis Detallado por Activo")
        ticker_buscar = st.selectbox("Selecciona un Ticker para auditar su fórmula:", sorted(df_clean["Ticker"].unique()))
        
        if ticker_buscar:
            row = df_clean[df_clean["Ticker"] == ticker_buscar].iloc[0]
            st.markdown(f"### Reporte de Valuación Estructural para **{ticker_buscar}**")
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.markdown(f"<div class='metric-card'><h4>Clasificación Algorítmica</h4><h2>{row['Clasificacion']}</h2></div>", unsafe_allow_html=True)
            with col_b2:
                st.markdown(f"<div class='metric-card'><h4>Probabilidad de Upside (90d)</h4><h2>{row['Probabilidad_Logit']*100:.2f}%</h2></div>", unsafe_allow_html=True)
            with col_b3:
                st.markdown(f"<div class='metric-card'><h4>P/E Ratio Coetáneo</h4><h2>{row['PE_Actual']:.2f}v</h2></div>", unsafe_allow_html=True)
            
            st.markdown("#### Desglose de Factores del Vector $X_j$ y Ponderaciones")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.write("**Métricas de Valuación Crítica (Fuerza del Algoritmo):**")
                st.info(f"📍 **Percentil Móvil P/E (24 Meses):** {row['Percentil_PE_24M']*100:.1f}% (0% es mínimo histórico; 100% techo de ciclo corto).")
                st.info(f"📍 **Z-Score del P/E:** {row['Z_Score_PE']:.2f} desviaciones estándar respecto a su media.")
                st.info(f"📍 **Free Cash Flow Yield:** {row['FCF_Yield']*100:.2f}% de rendimiento de efectivo disponible.")
            
            with col_f2:
                st.write("**Métricas de Calidad Estructural e Impulso:**")
                st.success(f"📈 **Margen Bruto:** {row['Margen_Bruto']*100:.1f}%")
                st.success(f"📈 **Aceleración Trimestral de Margen Bruto ($\Delta$):** {row['Delta_Margen_Bruto']*100:.2f}%")
                st.success(f"📈 **Eficiencia Sostenida (ROIC):** {row['ROIC']*100:.1f}%")
                st.info(f"📈 **Cobertura de Intereses de Deuda:** {row['Cobertura_Interes']:.1f}x")

    # PESTAÑA 3: LA LISTA NEGRA DE CONTROL DE RIESGOS (AUDITORÍA ACTUARIAL)
    with tab_auditoria:
        st.subheader("Filtro Sanitario: Empresas Expulsadas del Modelo")
        st.markdown("""
            El modelo Logit aplica un criterio estricto de admisión financiera antes de calcular múltiplos relativos. 
            Abajo se detallan los activos bloqueados y la **métrica exacta que provocó su exclusión** para asegurar la calidad de las entradas.
        """)
        
        if df_vetadas is not None and not df_vetadas.empty:
            razones_disponibles = ["Todas"] + list(df_vetadas["Razón"].unique())
            razon_sel = st.selectbox("Filtrar por Razón de Rechazo:", razones_disponibles)
            
            df_vetadas_filt = df_vetadas if razon_sel == "Todas" else df_vetadas[df_vetadas["Razón"] == razon_sel]
            st.dataframe(df_vetadas_filt, use_container_width=True, height=450, hide_index=True)
        else:
            st.info("Felicidades. Ningún activo del universo ha sido vetado bajo los parámetros sanitarios actuales.")

else:
    st.warning("⚠️ No se encontraron las bases de datos locales. Presiona el botón 'Forzar Ingesta y Recálculo Logit' en la barra lateral para poblar el modelo por primera vez.")
