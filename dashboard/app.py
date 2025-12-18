import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# --- CONFIGURATION DES CHEMINS ---
# On se base sur le dossier courant du fichier dashboard/app.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

LOGS_PATH = os.path.join(PROJECT_ROOT, "api", "production_logs", "api_request_log.jsonl")
DRIFT_REPORT_HTML = os.path.join(PROJECT_ROOT, "monitoring", "reports", "drift_report.html")

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Dashboard de Monitoring - Prêt à Dépenser",
    page_icon="📊",
    layout="wide"
)

st.title("🏦 Dashboard de Monitoring - Prêt à Dépenser")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_logs():
    if not os.path.exists(LOGS_PATH):
        return pd.DataFrame()
    
    data = []
    with open(LOGS_PATH, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

df_logs = load_logs()

# --- INTERFACE ---
tab1, tab2 = st.tabs(["🚀 Performance API (Ops)", "📉 Qualité des Données (Drift)"])

# === TAB 1 : PERFORMANCE OPÉRATIONNELLE ===
with tab1:
    st.header("Surveillance de l'API en Production")
    
    if df_logs.empty:
        st.warning("Aucun log disponible pour le moment.")
    else:
        # 1. KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        total_calls = len(df_logs)
        success_rate = (df_logs['status'] == 'SUCCESS').mean() * 100
        avg_latency = df_logs['latency_ms'].mean() if 'latency_ms' in df_logs.columns else 0
        errors = df_logs[df_logs['status'] == 'FAILURE'].shape[0]

        col1.metric("Appels Totaux", total_calls)
        col2.metric("Taux de Succès", f"{success_rate:.1f}%", delta_color="normal" if success_rate > 95 else "inverse")
        col3.metric("Latence Moyenne", f"{avg_latency:.1f} ms")
        col4.metric("Erreurs", errors, delta_color="inverse")

        st.divider()

        # 2. Graphiques
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("⏱️ Latence dans le temps")
            if 'timestamp' in df_logs.columns and 'latency_ms' in df_logs.columns:
                # Conversion timestamp
                df_logs['date'] = pd.to_datetime(df_logs['timestamp'], unit='s')
                fig_latency = px.line(df_logs, x='date', y='latency_ms', title="Évolution de la Latence (ms)")
                st.plotly_chart(fig_latency, use_container_width=True)

        with c2:
            st.subheader("Distribution des Décisions")
            if 'decision' in df_logs.columns:
                fig_pie = px.pie(df_logs, names='decision', title="Répartition Accord / Refus")
                st.plotly_chart(fig_pie, use_container_width=True)

        # 3. Tableau des logs récents
        st.subheader("Derniers Logs")
        st.dataframe(df_logs.tail(10).iloc[::-1]) # Inverser pour voir les derniers en premier

# === TAB 2 : DATA DRIFT ===
with tab2:
    st.header("Analyse de la Dérive des Données (Data Drift)")
    st.info("Ce rapport est généré par Evidently AI en comparant les logs de production avec les données d'entraînement.")
    
    if os.path.exists(DRIFT_REPORT_HTML):
        # On lit le fichier HTML généré par le script drift_analysis.py
        with open(DRIFT_REPORT_HTML, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # On l'intègre dans Streamlit
        components.html(html_content, height=1000, scrolling=True)
        
        # Bouton de rafraichissement (qui relancerait idéalement le script, ici simulé)
        if st.button("🔄 Rafraîchir l'analyse"):
            st.toast("Lancer 'python monitoring/drift_analysis.py' pour mettre à jour ce rapport.")
    else:
        st.error(f"Le rapport n'a pas été trouvé à l'emplacement : {DRIFT_REPORT_HTML}")
        st.write("👉 Avez-vous lancé le script `python monitoring/drift_analysis.py` ?")