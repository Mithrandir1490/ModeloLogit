import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página institucional
st.set_page_config(
    page_title="SBS Quant Lab - Motor Logit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para la sala de trading
st.markdown("""
    <style>
    .metric-card {
        background-color: #f7fafc;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2b6cb0;
        margin-bottom: 10px;
    }
    .top10-container {
        background-color: #fffaf0;
        padding: 20px;
        border-radius: 6px;
        border: 1px solid #feebc8;
        border-left: 6px solid #dd6b20;
        margin-bottom: 25px;
    }
    .ticker-badge {
        background-color: #2b6cb0;
        color: white;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: bold;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 SBS Center - Laboratorio Analítico de Valuación Logit")
st.markdown("---")

# Carga segura de las bases de datos optimizadas generadas por la nube
@st.cache_data
def cargar_datos():
    try:
        df_clean = pd.read_csv("logit_data.csv")
        df_vetadas = pd.read_csv("vetados_quality.csv")
        return df_clean, df_vetadas
    except Exception as e:
        st.error(f"Error al cargar las bases de datos: {e}")
        return None, None

df_clean, df_vetadas = cargar_datos()

if df_clean is not None:
    
    # ======================================================================
    # 🔥 SECCIÓN EXCLUSIVA: EL TOP 10 DE CONVICCIÓN CUANTITATIVA SBS
    # ======================================================================
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
        
        # Formatear la tabla del Top 10 para los usuarios finales
        df_top10_display = df_top10.copy()
        df_top10_display["Probabilidad_Logit"] = df_top10_display["Probabilidad_Logit"].map(lambda x: f"{x*100:.2f}%")
        df_top10_display["FCF_Yield"] = df_top10_display["FCF_Yield"].map(lambda x: f"{x*100:.2f}%")
        df_top10_display["Percentil_PE_24M"] = df_top10_display["Percentil_PE_24M"].map(lambda x: f"{x*100:.1f}%")
        df_top10_display["ROIC"] = df_top10_display["ROIC"].map(lambda x: f"{x*100:.1f}%")
        
        # Desplegar los 10 ganadores de forma simplificada y directa
        st.dataframe(df_top10_display[[
            "Ticker", "Clasificacion", "Probabilidad_Logit", "PE_Actual", 
            "Percentil_PE_24M", "FCF_Yield", "ROIC"
        ]], use_container_width=True, hide_index=True)
    else:
        st.info("🎯 El mercado se encuentra cotizando en rangos de valuación justos o elevados. No hay activos en zona de estrés o descuento que activen el Top 10 actualmente.")
        
    st.markdown("</div>", unsafe_allow_html=True)
    # ======================================================================

    # --- BARRA LATERAL: PANEL DE FILTROS ---
    st.sidebar.header("📊 Filtros de Exploración")
    categorias_disponibles = ["Todos"] + list(df_clean["Clasificacion"].unique())
    categoria_sel = st.sidebar.selectbox("Filtrar Universo General por Convicción:", categorias_disponibles)
    
    if categoria_sel != "Todos":
        df_filtrado = df_clean[df_clean["Clasificacion"] == categoria_sel]
    else:
        df_filtrado = df_clean

    # --- MÉTRICAS DE ESCALAFÓN GLOBAL ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Universo Analizado", len(df_clean) + len(df_vetadas))
    with col_m2:
        st.metric("Aprobadas por Calidad", len(df_clean), delta=f"{len(df_clean)/(len(df_clean)+len(df_vetadas))*100:.1f}%")
    with col_m3:
        gangas_count = len(df_clean[df_clean["Clasificacion"] == "💎 GANGA"])
        st.metric("GANGAS en el Radar", gangas_count, delta="Horizonte 90 Días", delta_color="inverse")
    with col_m4:
        st.metric("Vetadas por Riesgo", len(df_vetadas))

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
        df_display = df_filtrado.copy()
        df_display["Probabilidad_Logit"] = df_display["Probabilidad_Logit"].map(lambda x: f"{x*100:.2f}%")
        df_display["Percentil_PE_24M"] = df_display["Percentil_PE_24M"].map(lambda x: f"{x*100:.1f}%")
        df_display["FCF_Yield"] = df_display["FCF_Yield"].map(lambda x: f"{x*100:.2f}%")
        df_display["Margen_Bruto"] = df_display["Margen_Bruto"].map(lambda x: f"{x*100:.1f}%")
        df_display["ROIC"] = df_display["ROIC"].map(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(df_display[[
            "Ticker", "Clasificacion", "Probabilidad_Logit", "PE_Actual", 
            "Percentil_PE_24M", "Z_Score_PE", "FCF_Yield", "Margen_Bruto", "ROIC"
        ]], use_container_width=True, hide_index=True)

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
                st.info(f"📍 **Z-Score del P/E:** {row['Z_Score_PE']:.2f} desviaciones estándar.")
                st.info(f"📍 **Free Cash Flow Yield:** {row['FCF_Yield']*100:.2f}% de rendimiento de efectivo.")
            
            with col_f2:
                st.write("**Métricas de Calidad Estructural e Impulso:**")
                st.success(f"📈 **Margen Bruto:** {row['Margen_Bruto']*100:.1f}%")
                st.success(f"📈 **Aceleración Trimestral de Margen Bruto ($\Delta$):** {row['Delta_Margen_Bruto']*100:.2f}%")
                st.success(f"📈 **Eficiencia Sostenida (ROIC):** {row['ROIC']*100:.1f}%")
                st.success(f"📈 **Cobertura de Intereses de Deuda:** {row['Cobertura_Interes']:.1f}x")

    # PESTAÑA 3: LA LISTA NEGRA DE CONTROL DE RIESGOS (AUDITORÍA ACTUARIAL)
    with tab_auditoria:
        st.subheader("Filtro Sanitario: Empresas Expulsadas del Modelo")
        st.markdown("""
            El modelo Logit aplica un criterio estricto de admisión financiera antes de calcular múltiplos relativos. 
            Abajo se detallan los activos bloqueados y la **métrica exacta que provocó su exclusión** para asegurar la calidad de las entradas.
        """)
        
        razones_disponibles = ["Todas"] + list(df_vetadas["Razón"].unique())
        razon_sel = st.selectbox("Filtrar por Razón de Rechazo:", razones_disponibles)
        
        if razon_sel != "Todas":
            df_vetadas_filt = df_vetadas[df_vetadas["Razón"] == razon_sel]
        else:
            df_vetadas_filt = df_vetadas
            
        st.dataframe(df_vetadas_filt, use_container_width=True, hide_index=True)

else:
    st.warning("⚠️ No se encontraron las bases de datos. El Action en GitHub debe completar su ejecución para poblar el modelo.")
