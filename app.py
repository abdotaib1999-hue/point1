import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Évaluation Sismique - Méthode N2 (RPA 2024)", layout="wide")

# ==========================================
# FONCTIONS MÉTIER (LOGIQUE STRUCTURELLE)
# ==========================================

def get_rpa_params(zone, soil_type):
    """Récupère les paramètres du RPA 2024 selon la zone et le site."""
    # Coefficients d'accélération A
    A_dict = {"I": 0.07, "II": 0.10, "III": 0.15, "IV": 0.20, "V": 0.25, "VI": 0.30}
    A = A_dict[zone]
    
    # Choix du Type de Spectre (1 pour zones fortes, 2 pour zones faibles/moyennes)
    type_spectre = 1 if zone in ["IV", "V", "VI"] else 2

    # Paramètres de site (S, T1, T2, T3)
    if type_spectre == 1:
        params = {
            "S1": (1.00, 0.10, 0.40, 2.0),
            "S2": (1.20, 0.10, 0.50, 2.0),
            "S3": (1.30, 0.15, 0.60, 2.0),
            "S4": (1.35, 0.15, 0.70, 2.0)
        }
    else:
        params = {
            "S1": (1.00, 0.05, 0.25, 1.20),
            "S2": (1.30, 0.05, 0.30, 1.20),
            "S3": (1.55, 0.10, 0.40, 1.20),
            "S4": (1.80, 0.10, 0.50, 1.20)
        }
        
    S, T1, T2, T3 = params[soil_type]
    return A, S, T1, T2, T3

def calc_Sae(T, A, I_factor, S, T1, T2, T3, damping):
    """Calcule l'accélération spectrale élastique Sae (RPA 2024) en [g]."""
    eta = np.sqrt(7.0 / (2.0 + damping))
    if T < T1:
        return A * I_factor * S * (1 + (T/T1) * (2.5 * eta - 1))
    elif T < T2:
        return A * I_factor * S * 2.5 * eta
    elif T < T3:
        return A * I_factor * S * 2.5 * eta * (T2 / T)
    else:
        return A * I_factor * S * 2.5 * eta * (T2 * T3) / (T**2)

def bilinearize_pushover(d_star, F_star):
    """
    Idéalisation bilinéaire de la courbe de capacité (SDOF) basée sur l'égalité des énergies.
    """
    dm_star = d_star[-1]
    
    # Énergie de déformation (aire sous la courbe)
    Em_star = np.trapz(F_star, d_star)
    
    # Calcul de la rigidité initiale sécante à 60% de F_max
    F_max = np.max(F_star)
    idx_06 = np.where(F_star >= 0.6 * F_max)[0]
    if len(idx_06) == 0:
        idx_06 = [1]
    idx = max(1, idx_06[0])
    
    K_star = F_star[idx] / d_star[idx]
    
    # Résolution de l'équation du second degré pour l'égalité d'énergie
    # 0.5 * (1/K) * Fy^2 - dm * Fy + Em = 0
    a = 0.5 / K_star
    b = -dm_star
    c = Em_star
    delta = b**2 - 4*a*c
    
    if delta >= 0:
        Fy_star = (-b - np.sqrt(delta)) / (2*a)
    else:
        Fy_star = F_max  # Fallback si courbe atypique
        
    dy_star = Fy_star / K_star
    return K_star, Fy_star, dy_star

def method_n2(d_array, V_array, Gamma, m_star, A, I_factor, S, T1, T2, T3, damping):
    """Application de la méthode N2 pour trouver le point de performance."""
    # 1. Conversion MDOF -> SDOF
    d_star = d_array / Gamma
    F_star = V_array / Gamma
    
    # 2. Idéalisation bilinéaire
    K_star, Fy_star, dy_star = bilinearize_pushover(d_star, F_star)
    
    # 3. Période du système SDOF équivalent
    T_star = 2 * np.pi * np.sqrt(m_star * dy_star / (Fy_star * 1000)) # F en kN, m_star en Tonnes, d en m
    
    # 4. Demande Sismique Elastique
    Sae_g = calc_Sae(T_star, A, I_factor, S, T1, T2, T3, damping)
    Sae_ms2 = Sae_g * 9.81
    
    # Déplacement élastique cible
    det_star = Sae_ms2 * (T_star / (2 * np.pi))**2
    
    # Force élastique correspondante
    Fet_star = m_star * Sae_ms2 / 1000 # [kN]
    
    # Coefficient de réduction q_u (ou R_mu)
    q_u = Fet_star / Fy_star
    
    # 5. Déplacement cible SDOF inélastique
    if T_star < T2: # Domaine des courtes périodes
        if q_u <= 1:
            dt_star = det_star
        else:
            dt_star = (det_star / q_u) * (1 + (q_u - 1) * (T2 / T_star))
    else: # Domaine des périodes moyennes et longues (Règle des déplacements égaux)
        dt_star = det_star
        
    # 6. Re-conversion SDOF -> MDOF
    d_t = Gamma * dt_star
    
    # Interpolation de l'effort tranchant MDOF correspondant
    V_t = np.interp(d_t, d_array, V_array)
    
    return {
        "d_star": d_star, "F_star": F_star,
        "dy_star": dy_star, "Fy_star": Fy_star,
        "K_star": K_star, "T_star": T_star,
        "Sae_g": Sae_g, "det_star": det_star,
        "q_u": q_u, "dt_star": dt_star,
        "d_t": d_t, "V_t": V_t
    }

# ==========================================
# INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================

st.title("🏗️ Analyse Pushover & Méthode N2 (RPA 2024)")
st.markdown("""
Cette application détermine le **Point de Performance** d'une structure à partir de sa courbe de capacité (Pushover) 
selon la méthode N2, en intégrant le spectre de réponse élastique du **Règlement Parasismique Algérien (RPA 2024)**.
""")

with st.sidebar:
    st.header("1. Fichier de Capacité")
    uploaded_file = st.file_uploader("Importer CSV ou Excel", type=["csv", "xlsx", "xls"])
    unite_d = st.selectbox("Unité de déplacement du fichier", ["m", "cm", "mm"], index=1)
    
    st.header("2. Paramètres Sismiques (RPA 2024)")
    zone = st.selectbox("Zone Sismique", ["I", "II", "III", "IV", "V", "VI"], index=4)
    soil = st.selectbox("Type de Sol", ["S1", "S2", "S3", "S4"], index=1)
    
    importance_dict = {"1A (Vital)": 1.4, "1B (Très important)": 1.2, "2 (Courant)": 1.0, "3 (Faible)": 0.8}
    importance_key = st.selectbox("Groupe d'importance", list(importance_dict.keys()), index=2)
    importance_factor = importance_dict[importance_key]
    
    damping = st.number_input("Amortissement visqueux (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
    
    st.header("3. Paramètres Structuraux")
    st.info("💡 Γ est le facteur de participation modale. $m^*$ est la masse modale effective (en Tonnes).")
    mass_total = st.number_input("Poids total de la structure W (kN)", value=15000.0, step=100.0)
    Gamma = st.number_input("Facteur de transformation (Γ)", value=1.25, step=0.05)
    ratio_m = st.number_input("Ratio masse équivalente (m* / M_tot)", value=0.8, step=0.05, max_value=1.0)
    
if uploaded_file is not None:
    # --- Traitement des données ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        if df.shape[1] < 2:
            st.error("Le fichier doit contenir au moins deux colonnes (Déplacement et Effort Tranchant).")
            st.stop()
            
        # Nettoyage
        df = df.iloc[:, :2] # On prend les 2 premières colonnes
        df.columns = ["Deplacement", "BaseShear"]
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df[df["Deplacement"].diff().fillna(1) >= 0] # Forcer la monotonicité
        
        # Conversion en mètres
        if unite_d == "cm":
            df["Deplacement_m"] = df["Deplacement"] / 100.0
        elif unite_d == "mm":
            df["Deplacement_m"] = df["Deplacement"] / 1000.0
        else:
            df["Deplacement_m"] = df["Deplacement"]
            
        d_array = df["Deplacement_m"].values
        V_array = df["BaseShear"].values
        
        # --- Calculs N2 ---
        M_tot = mass_total / 9.81
        m_star = M_tot * ratio_m
        A, S, T1, T2, T3 = get_rpa_params(zone, soil)
        
        res = method_n2(d_array, V_array, Gamma, m_star, A, importance_factor, S, T1, T2, T3, damping)
        
        # --- Affichage des Résultats ---
        st.markdown("---")
        st.subheader("📊 Résultats de l'Évaluation Sismique")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Période équivalente T*", f"{res['T_star']:.2f} s")
        col2.metric("Déplacement cible (Toit)", f"{res['d_t']*100:.1f} cm")
        col3.metric("Effort au pt de performance", f"{res['V_t']:.0f} kN")
        col4.metric("Ductilité demandée (\u03bc)", f"{(res['d_t']/Gamma)/res['dy_star']:.2f}")

        # --- Graphiques Plotly ---
        tab1, tab2 = st.tabs(["📉 Courbe de Capacité (MDOF)", "🎯 Format ADRS (Spectre de Capacité)"])
        
        with tab1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=d_array*100, y=V_array, mode='lines', name='Courbe Pushover', line=dict(color='blue', width=3)))
            # Courbe bilinéaire projetée MDOF
            d_bili_mdof = np.array([0, res['dy_star']*Gamma, d_array[-1]]) * 100
            V_bili_mdof = np.array([0, res['Fy_star']*Gamma, res['Fy_star']*Gamma])
            fig1.add_trace(go.Scatter(x=d_bili_mdof, y=V_bili_mdof, mode='lines', name='Idéalisée bilinéaire', line=dict(color='red', dash='dash')))
            fig1.add_trace(go.Scatter(x=[res['d_t']*100], y=[res['V_t']], mode='markers', name='Point de Performance', marker=dict(color='orange', size=12, symbol='star')))
            
            fig1.update_layout(title="Courbe de Capacité de la Structure (Système Réel)",
                               xaxis_title="Déplacement au sommet (cm)", yaxis_title="Effort Tranchant à la base (kN)",
                               template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            # Construction des spectres pour l'ADRS
            T_vals = np.linspace(0.01, 4.0, 400)
            Sa_g_vals = [calc_Sae(t, A, importance_factor, S, T1, T2, T3, damping) for t in T_vals]
            Sa_ms2_vals = np.array(Sa_g_vals) * 9.81
            Sd_m_vals = Sa_ms2_vals * (T_vals / (2*np.pi))**2
            
            fig2 = go.Figure()
            # Demande (Spectre élastique)
            fig2.add_trace(go.Scatter(x=Sd_m_vals, y=Sa_g_vals, mode='lines', name='Spectre de Demande (RPA)', line=dict(color='green', width=2)))
            
            # Capacité SDOF
            Sa_cap_g = res['F_star'] / (m_star * 9.81)
            fig2.add_trace(go.Scatter(x=res['d_star'], y=Sa_cap_g, mode='lines', name='Spectre de Capacité', line=dict(color='blue', width=3)))
            
            # Bilinéaire SDOF
            Sa_bili_g = np.array([0, res['Fy_star'], res['Fy_star']]) / (m_star * 9.81)
            d_bili_star = np.array([0, res['dy_star'], res['d_star'][-1]])
            fig2.add_trace(go.Scatter(x=d_bili_star, y=Sa_bili_g, mode='lines', name='Capacité Idéalisée', line=dict(color='red', dash='dash')))
            
            # Point de performance
            fig2.add_trace(go.Scatter(x=[res['dt_star']], y=[res['Sae_g'] if res['T_star'] >= T2 else (res['Fy_star']/(m_star*9.81))], 
                                      mode='markers', name='Point Cible', marker=dict(color='orange', size=12, symbol='star')))
            
            # Ligne de période élastique (T*)
            max_d_line = max(Sd_m_vals)
            max_a_line = max_d_line * (2*np.pi/res['T_star'])**2 / 9.81
            fig2.add_trace(go.Scatter(x=[0, max_d_line], y=[0, max_a_line], mode='lines', name=f'Période T*={res["T_star"]:.2f}s', line=dict(color='gray', dash='dot')))
            
            fig2.update_layout(title="Format ADRS (Accélération - Déplacement)",
                               xaxis_title="Déplacement Spectral Sd (m)", yaxis_title="Accélération Spectrale Sa (g)",
                               template="plotly_white", xaxis=dict(range=[0, min(max(res['d_star'])*1.2, max(Sd_m_vals))]), 
                               yaxis=dict(range=[0, max(max(Sa_cap_g)*1.2, max(Sa_g_vals))]))
            st.plotly_chart(fig2, use_container_width=True)

        # --- Tableau et Export ---
        st.markdown("### 📋 Tableau de synthèse")
        df_results = pd.DataFrame({
            "Paramètre": ["Période T* (s)", "Force plastification F_y* (kN)", "Déplac. élastique d_y* (m)", 
                          "Déplac. Cible SDOF d_t* (m)", "Facteur de réduction q_u", "Déplac. Cible Toit MDOF d_t (cm)"],
            "Valeur": [f"{res['T_star']:.3f}", f"{res['Fy_star']:.1f}", f"{res['dy_star']:.4f}",
                       f"{res['dt_star']:.4f}", f"{res['q_u']:.2f}", f"{res['d_t']*100:.2f}"]
        })
        st.table(df_results)
        
        # Export CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Exporter le résumé (CSV)", data=csv, file_name="N2_Resultats_RPA2024.csv", mime="text/csv")
        
    except Exception as e:
        st.error(f"⚠️ Erreur lors du traitement : {str(e)}")
        st.info("Vérifiez que votre fichier contient bien 2 colonnes (Déplacement en 1er, Effort en 2ème) avec des valeurs numériques.")

else:
    st.info("👈 Veuillez importer un fichier Pushover dans la barre latérale pour commencer.")
