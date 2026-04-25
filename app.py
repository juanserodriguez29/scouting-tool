import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Scouting Tool",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --verde:   #00E676;
    --oscuro:  #0A0E14;
    --gris1:   #111827;
    --gris2:   #1F2937;
    --gris3:   #374151;
    --texto:   #F9FAFB;
    --apagado: #9CA3AF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--oscuro);
    color: var(--texto);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--gris1);
    border-right: 1px solid var(--gris3);
}

/* Header título */
.titulo-app {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 4px;
    color: var(--verde);
    line-height: 1;
    margin-bottom: 4px;
}
.subtitulo-app {
    font-size: 0.85rem;
    color: var(--apagado);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    color: var(--apagado) !important;
    text-transform: uppercase !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--verde) !important;
    border-bottom: 2px solid var(--verde) !important;
}

/* Cards de estadísticas */
.stat-card {
    background: var(--gris1);
    border: 1px solid var(--gris3);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stat-card .valor {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--verde);
    line-height: 1;
}
.stat-card .etiqueta {
    font-size: 0.72rem;
    color: var(--apagado);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* Badge posición */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-GK  { background: #1e3a5f; color: #60A5FA; }
.badge-DEF { background: #14532d; color: #4ADE80; }
.badge-MID { background: #4a1942; color: #E879F9; }
.badge-FWD { background: #7c2d12; color: #FB923C; }

/* Jugador similar card */
.similar-card {
    background: var(--gris1);
    border: 1px solid var(--gris3);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.similar-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: var(--gris3);
    width: 40px;
}
.similar-info { flex: 1; padding: 0 12px; }
.similar-name { font-weight: 600; font-size: 1rem; }
.similar-meta { font-size: 0.78rem; color: var(--apagado); }
.similar-dist {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    color: var(--verde);
    text-align: right;
}
.similar-dist small { font-family: 'DM Sans', sans-serif; font-size: 0.65rem; color: var(--apagado); display: block; }

/* Botones */
.stButton > button {
    background: var(--verde) !important;
    color: var(--oscuro) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
}
.stButton > button:hover {
    background: #00c853 !important;
    transform: translateY(-1px);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--gris3) !important;
    border-radius: 10px !important;
}

/* Selectbox / multiselect */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {
    color: var(--apagado) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Separador */
hr { border-color: var(--gris3) !important; }

/* Alerta de datos faltantes */
.aviso {
    background: #1a1200;
    border: 1px solid #d97706;
    border-radius: 10px;
    padding: 16px 20px;
    color: #fbbf24;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LABELS Y MAPAS
# ─────────────────────────────────────────────
BADGE_CLASS = {
    'Goalkeepers': 'badge-GK',
    'Defenders':   'badge-DEF',
    'Midfielders': 'badge-MID',
    'Forwards':    'badge-FWD',
}

LABEL_ES = {
    'Goalkeepers': 'Porteros',
    'Defenders':   'Defensas',
    'Midfielders': 'Mediocampistas',
    'Forwards':    'Delanteros',
}

FEATURES_LABEL = {
    'goals_p90':                    'Goles p90',
    'assists_p90':                  'Asistencias p90',
    'shots_p90':                    'Disparos p90',
    'keyPasses_p90':                'Pases clave p90',
    'interceptions_p90':            'Intercepciones p90',
    'groundDuelsWon_p90':           'Duelos en tierra ganados p90',
    'totalDuelsWon_p90':            'Duelos totales ganados p90',
    'dribbles_p90':                 'Regates p90',
    'bigChancesCreated_p90':        'Grandes chances creadas p90',
    'accurateFinalThirdPasses_p90': 'Pases al último tercio p90',
    'accuratePasses_p90':           'Pases precisos p90',
    'wasFouled_p90':                'Faltas recibidas p90',
    'accurateLongBalls_p90':        'Balones largos precisos p90',
    'fouls_p90':                    'Faltas cometidas p90',
    'dribbledPast_p90':             'Regateado p90',
    'saves_p90':                    'Atajadas p90',
    'accuratePassesPercentage':     '% Pases precisos',
    'groundDuelsWonPercentage':     '% Duelos en tierra ganados',
    'totalDuelsWonPercentage':      '% Duelos totales ganados',
    'accurateLongBallsPercentage':  '% Balones largos precisos',
    'savePercentage':               '% Atajadas',
    'cleanSheet':                   'Vallas invictas',
    'goalsConcededPerGame':         'Goles encajados por partido',
}

LIGA_LABEL = {
    'col': '🇨🇴 Colombia Primera A',
    'arg': '🇦🇷 Argentina Liga Profesional',
    'bra': '🇧🇷 Brasileirão Série A',
    'bol': '🇧🇴 Bolivia División Profesional',
    'chi': '🇨🇱 Chile Primera División',
    'ecu': '🇪🇨 Ecuador LigaPro',
    'per': '🇵🇪 Peru Liga 1',
    'uru': '🇺🇾 Uruguay Primera División',
    'ven': '🇻🇪 Venezuela Primera División',
    'mls': '🇺🇸 MLS',
}

# ─────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────
DATA_PATH    = 'data/jugadores_2026.csv'
SCALERS_PATH = 'models/scalers.pkl'
KMEANS_PATH  = 'models/kmeans_models.pkl'
FEAT_PATH    = 'models/features_por_posicion.pkl'

@st.cache_data(show_spinner=False)
def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource(show_spinner=False)
def cargar_modelos():
    with open(SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    with open(KMEANS_PATH, 'rb') as f:
        kmeans_models = pickle.load(f)
    with open(FEAT_PATH, 'rb') as f:
        features_por_pos = pickle.load(f)
    return scalers, kmeans_models, features_por_pos

datos_ok = os.path.exists(DATA_PATH) and os.path.exists(SCALERS_PATH)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="titulo-app">⚽ SCOUTING TOOL</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo-app">Análisis de jugadores · Sudamérica & MLS · 2026</div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — ACTUALIZAR DATOS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔄 Datos")

    if datos_ok:
        try:
            df_check = pd.read_csv(DATA_PATH, usecols=['player'])
            st.success(f"**{len(df_check):,}** jugadores cargados")
        except:
            st.success("Datos disponibles")
    else:
        st.markdown('<div class="aviso">⚠️ No hay datos todavía.</div>', unsafe_allow_html=True)

    st.markdown("")

    # Detectar si estamos en Streamlit Cloud
    EN_LA_NUBE = os.environ.get("HOME", "") == "/home/appuser"

    if EN_LA_NUBE:
        st.markdown("""
        <div style="background:#0f1f0f;border:1px solid #166534;border-radius:10px;padding:14px 16px;">
            <div style="font-size:0.78rem;color:#4ADE80;font-weight:600;margin-bottom:6px;">📡 ACTUALIZAR DATOS</div>
            <div style="font-size:0.78rem;color:#9CA3AF;line-height:1.6;">
                Corre el notebook localmente y sube los archivos a GitHub:
                <br><br>
                <code style="background:#1F2937;padding:2px 6px;border-radius:4px;font-size:0.72rem;">git add data/ models/</code><br><br>
                <code style="background:#1F2937;padding:2px 6px;border-radius:4px;font-size:0.72rem;">git commit -m "actualizar datos"</code><br><br>
                <code style="background:#1F2937;padding:2px 6px;border-radius:4px;font-size:0.72rem;">git push</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("🔄 Actualizar datos desde Sofascore", use_container_width=True):
            with st.spinner("Corriendo el notebook... Esto puede tomar varios minutos ⏳"):
                try:
                    result = subprocess.run(
                        ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                         "--ExecutePreprocessor.timeout=1800",
                         "KMeans_reemplazo_jugadores.ipynb",
                         "--output", "KMeans_reemplazo_jugadores_ejecutado.ipynb"],
                        capture_output=True, text=True, timeout=1900
                    )
                    if result.returncode == 0:
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success("✅ Datos actualizados. Recarga la página.")
                        st.balloons()
                    else:
                        st.error(f"Error:\n{result.stderr[-1000:]}")
                except subprocess.TimeoutExpired:
                    st.error("El scraping tardó demasiado. Intenta desde el notebook.")
                except Exception as e:
                    st.error(f"Error inesperado: {e}")

    st.markdown("---")
    st.markdown('<span style="font-size:0.75rem;color:#6B7280;">Desarrollado con LanusStats + KMeans</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN — solo si hay datos
# ─────────────────────────────────────────────
if not datos_ok:
    st.markdown("""
    <div style="text-align:center;padding:80px 0;">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:3rem;color:#374151;">SIN DATOS</div>
        <div style="color:#6B7280;margin-top:8px;">Usa el botón del panel lateral para actualizar los datos desde Sofascore.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Cargar
df = cargar_datos()
scalers, kmeans_models, FEATURES_POR_POSICION = cargar_modelos()

# Stats globales
col1, col2, col3, col4 = st.columns(4)
stat_cards = [
    (len(df), "Jugadores"),
    (df['position_group'].nunique(), "Posiciones"),
    (df['league_name'].nunique(), "Ligas"),
    (df['team'].nunique() if 'team' in df.columns else "—", "Equipos"),
]
for col, (val, label) in zip([col1, col2, col3, col4], stat_cards):
    with col:
        st.markdown(f'<div class="stat-card"><div class="valor">{val}</div><div class="etiqueta">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🏆  Ranking de Jugadores", "🔍  Buscar Similar"])

# ══════════════════════════════════════════════
# TAB 1 — RANKING
# ══════════════════════════════════════════════
with tab1:
    st.markdown("#### Filtros")

    f1, f2, f3 = st.columns(3)

    with f1:
        posicion_sel = st.selectbox(
            "Posición",
            options=list(FEATURES_POR_POSICION.keys()),
            format_func=lambda x: LABEL_ES[x]
        )

    ligas_disponibles = sorted(df['league_code'].unique())
    ligas_opciones = {k: v for k, v in LIGA_LABEL.items() if k in ligas_disponibles}

    with f2:
        ligas_sel = st.multiselect(
            "Ligas",
            options=list(ligas_opciones.keys()),
            default=list(ligas_opciones.keys()),
            format_func=lambda x: ligas_opciones.get(x, x)
        )

    with f3:
        min_min = st.number_input("Mínimo de minutos jugados", min_value=0, max_value=3000, value=600, step=90)

    # Features disponibles para la posición elegida
    feats_pos = [f for f in FEATURES_POR_POSICION[posicion_sel] if f in df.columns]
    feat_labels = [FEATURES_LABEL.get(f, f) for f in feats_pos]

    metrica_label = st.selectbox("Ordenar por métrica", options=feat_labels)
    metrica_col   = feats_pos[feat_labels.index(metrica_label)]

    top_n = st.slider("Mostrar top N jugadores", min_value=5, max_value=50, value=20, step=5)

    # Aplicar filtros
    df_filtered = df[
        (df['position_group'] == posicion_sel) &
        (df['league_code'].isin(ligas_sel)) &
        (df['minutesPlayed'] >= min_min)
    ].copy()

    if df_filtered.empty:
        st.warning("No hay jugadores con esos filtros. Intenta ampliar los criterios.")
    else:
        df_filtered = df_filtered.dropna(subset=[metrica_col])
        df_top = df_filtered.nlargest(top_n, metrica_col).reset_index(drop=True)
        df_top.index += 1

        # Columnas a mostrar
        cols_show = ['player', 'team', 'league_name', 'minutesPlayed', metrica_col]
        cols_extra = [f for f in feats_pos if f != metrica_col and f in df_top.columns][:4]
        cols_show += cols_extra

        cols_show = [c for c in cols_show if c in df_top.columns]

        rename_map = {
            'player':       'Jugador',
            'team':         'Equipo',
            'league_name':  'Liga',
            'minutesPlayed':'Minutos',
            metrica_col:    f'⭐ {metrica_label}',
        }
        for c in cols_extra:
            rename_map[c] = FEATURES_LABEL.get(c, c)

        df_display = df_top[cols_show].rename(columns=rename_map)

        # Formato numérico
        num_cols = [c for c in df_display.columns if df_display[c].dtype in [np.float64, np.float32]]
        df_display[num_cols] = df_display[num_cols].round(3)

        st.markdown(f"<br>**{len(df_filtered):,}** jugadores encontrados · mostrando top **{top_n}** por _{metrica_label}_", unsafe_allow_html=True)

        st.dataframe(
            df_display,
            use_container_width=True,
            height=min(50 + top_n * 36, 600),
        )

        # Download
        csv = df_top[cols_show].rename(columns=rename_map).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar tabla CSV",
            data=csv,
            file_name=f"ranking_{posicion_sel}_{metrica_col}.csv",
            mime='text/csv',
        )

# ══════════════════════════════════════════════
# TAB 2 — JUGADOR SIMILAR
# ══════════════════════════════════════════════
with tab2:
    st.markdown("#### Buscar jugador similar")
    st.markdown('<span style="font-size:0.8rem;color:#6B7280;">Los filtros van en cascada: selecciona posición → liga → equipo → jugador.</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fila 1: Posición + Liga
    sc1, sc2 = st.columns(2)

    with sc1:
        pos_sim = st.selectbox(
            "1 · Posición del jugador",
            options=list(FEATURES_POR_POSICION.keys()),
            format_func=lambda x: LABEL_ES[x],
            key="pos_sim"
        )

    # Filtrar por posición para ligas disponibles
    df_por_pos = df[df['position_group'] == pos_sim].copy()
    ligas_con_pos = sorted(df_por_pos['league_code'].dropna().unique())
    ligas_opciones_sim = {k: v for k, v in LIGA_LABEL.items() if k in ligas_con_pos}

    with sc2:
        liga_sim = st.selectbox(
            "2 · Liga",
            options=list(ligas_opciones_sim.keys()),
            format_func=lambda x: ligas_opciones_sim.get(x, x),
            key="liga_sim"
        )

    # ── Fila 2: Equipo + Jugador
    sc3, sc4 = st.columns(2)

    # Filtrar por posición + liga para equipos disponibles
    df_por_liga = df_por_pos[df_por_pos['league_code'] == liga_sim].copy()
    equipos_disponibles = sorted(df_por_liga['team'].dropna().unique()) if 'team' in df_por_liga.columns else []

    with sc3:
        equipo_sim = st.selectbox(
            "3 · Equipo",
            options=["— Todos los equipos —"] + equipos_disponibles,
            key="equipo_sim"
        )

    # Filtrar por equipo si se seleccionó uno específico
    if equipo_sim == "— Todos los equipos —":
        df_por_equipo = df_por_liga.copy()
    else:
        df_por_equipo = df_por_liga[df_por_liga['team'] == equipo_sim].copy()

    jugadores_disponibles = sorted(df_por_equipo['player'].dropna().unique())

    with sc4:
        if not jugadores_disponibles:
            st.selectbox("4 · Jugador de referencia", options=["Sin jugadores disponibles"], key="jugador_sel")
            jugador_sel = None
        else:
            jugador_sel = st.selectbox(
                "4 · Jugador de referencia",
                options=jugadores_disponibles,
                key="jugador_sel"
            )

    # ── Fila 3: opciones de búsqueda
    st.markdown("<br>", unsafe_allow_html=True)
    so1, so2 = st.columns(2)

    with so1:
        top_sim = st.slider("Número de jugadores similares a mostrar", 5, 20, 10, key="top_sim")

    with so2:
        # Filtro opcional de ligas en los RESULTADOS (los similares pueden venir de cualquier liga)
        ligas_resultado = st.multiselect(
            "Buscar similares solo en estas ligas (opcional)",
            options=list(ligas_opciones_sim.keys()),
            default=[],
            format_func=lambda x: ligas_opciones_sim.get(x, x),
            key="ligas_resultado",
            placeholder="Todas las ligas"
        )

    if st.button("🔍 Buscar similares", key="btn_similar", disabled=(jugador_sel is None)):
        feats_sim = [f for f in FEATURES_POR_POSICION[pos_sim] if f in df.columns]

        if pos_sim not in scalers:
            st.error(f"No hay modelo entrenado para {pos_sim}. Actualiza los datos.")
        else:
            scaler_sim = scalers[pos_sim]

            # Pool completo de la posición (la búsqueda de similares no se limita al equipo/liga del jugador)
            df_pos_all = df[df['position_group'] == pos_sim].copy()
            df_pos_clean = df_pos_all.dropna(subset=feats_sim).copy()

            # Aplicar filtro de ligas en resultados si se eligió alguna
            if ligas_resultado:
                df_candidates = df_pos_clean[df_pos_clean['league_code'].isin(ligas_resultado)].copy()
            else:
                df_candidates = df_pos_clean.copy()

            target = df_pos_clean[df_pos_clean['player'] == jugador_sel]

            if target.empty:
                st.warning(f"No se encontró a **{jugador_sel}** con datos suficientes en las métricas de {LABEL_ES[pos_sim]}.")
            else:
                target_row = target.iloc[[0]]
                others     = df_candidates[df_candidates['player'] != jugador_sel].copy()

                target_scaled = scaler_sim.transform(target_row[feats_sim])
                others_scaled  = scaler_sim.transform(others[feats_sim])

                dists = euclidean_distances(target_scaled, others_scaled)[0]
                others = others.copy()
                others['_distancia'] = dists
                similares = others.nsmallest(top_sim, '_distancia').reset_index(drop=True)

                # Info del jugador de referencia
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="similar-card" style="border-color:#00E676;margin-bottom:20px;">
                    <div class="similar-rank">#</div>
                    <div class="similar-info">
                        <div class="similar-name">🎯 {jugador_sel} <span style="font-weight:300;color:#9CA3AF;">(referencia)</span></div>
                        <div class="similar-meta">{target_row['team'].values[0] if 'team' in target_row else ''} · {target_row['league_name'].values[0] if 'league_name' in target_row else ''} · {int(target_row['minutesPlayed'].values[0])} min</div>
                    </div>
                    <div class="similar-dist">—<small>distancia</small></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**Top {top_sim} jugadores más similares a {jugador_sel}:**")

                for i, row in similares.iterrows():
                    team     = row['team'] if 'team' in row else ''
                    liga     = row['league_name'] if 'league_name' in row else ''
                    minutos  = int(row['minutesPlayed']) if 'minutesPlayed' in row else '—'
                    distancia = round(row['_distancia'], 3)

                    st.markdown(f"""
                    <div class="similar-card">
                        <div class="similar-rank">{i+1}</div>
                        <div class="similar-info">
                            <div class="similar-name">{row['player']}</div>
                            <div class="similar-meta">{team} · {liga} · {minutos} min</div>
                        </div>
                        <div class="similar-dist">{distancia}<small>distancia</small></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Tabla comparativa
                st.markdown("<br>**Comparativa de métricas:**", unsafe_allow_html=True)
                cols_comp = ['player'] + feats_sim
                df_comp = pd.concat([target_row[cols_comp], similares[cols_comp]], ignore_index=True)
                df_comp = df_comp.rename(columns={'player': 'Jugador', **{f: FEATURES_LABEL.get(f, f) for f in feats_sim}})
                num_c = [c for c in df_comp.columns if df_comp[c].dtype in [np.float64, np.float32]]
                df_comp[num_c] = df_comp[num_c].round(3)

                # Resaltar fila de referencia
                def highlight_ref(row):
                    if row.name == 0:
                        return ['background-color: #052e1a; color: #00E676; font-weight:700'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_comp.style.apply(highlight_ref, axis=1),
                    use_container_width=True,
                    height=min(100 + (top_sim+1) * 36, 600),
                )

                csv2 = df_comp.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar comparativa CSV",
                    data=csv2,
                    file_name=f"similares_{jugador_sel.replace(' ','_')}.csv",
                    mime='text/csv',
                    key="dl_sim"
                )
