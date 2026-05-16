import streamlit as st
import pandas as pd
import joblib
import math

# 1. Configuração da Página (Alterado para "wide" para expandir a horizontal!)
st.set_page_config(page_title="AI Football Predictor", page_icon="⚽", layout="wide")

# 2. Carregar os TRÊS Cérebros e os Dados
@st.cache_data
def carregar_dados():
    modelo_1x2 = joblib.load('modelo_xgboost.pkl')
    modelo_golos = joblib.load('modelo_xgboost_golos.pkl')
    modelo_btts = joblib.load('modelo_xgboost_btts.pkl')
    colunas = joblib.load('colunas.pkl')
    df = pd.read_csv('data/base_processada.csv')
    
    equipas = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
    return modelo_1x2, modelo_golos, modelo_btts, colunas, df, equipas

modelo_1x2, modelo_golos, modelo_btts, colunas, df, equipas = carregar_dados()

# Função Matemática de Poisson
def calcular_poisson(lam, k):
    return math.exp(-lam) * (lam**k) / math.factorial(k)

# 3. Interface Visual
st.title("⚽ Primeira Liga AI Predictor")
st.markdown("Previsões Quantitativas baseadas em **Sistema Elo, Form e Expected Goals (xG)**.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    casa = st.selectbox("🏟️ Equipa da Casa", equipas, index=equipas.index('Porto') if 'Porto' in equipas else 0)
with col2:
    fora = st.selectbox("✈️ Equipa de Fora", equipas, index=equipas.index('Benfica') if 'Benfica' in equipas else 1)

st.write("") 

if st.button("Gerar Previsão Total 🔮", use_container_width=True):
    if casa == fora:
        st.error("⚠️ As equipas da Casa e de Fora têm de ser diferentes!")
    else:
        with st.spinner('A consultar os três modelos de IA...'):
            
            ultimo_jogo_casa = df[(df['HomeTeam'] == casa) | (df['AwayTeam'] == casa)].iloc[-1]
            ultimo_jogo_fora = df[(df['HomeTeam'] == fora) | (df['AwayTeam'] == fora)].iloc[-1]
            
            def get_stat(row, equipa, stat_base):
                prefixo = "Home" if row['HomeTeam'] == equipa else "Away"
                return row[f"{prefixo}{stat_base}"]

            jogo_novo = pd.DataFrame(columns=colunas)
            jogo_novo.loc[0] = 0.0 
            
            for col in colunas:
                if 'Home' in col and col != 'H2H_Pontos_Casa':
                    stat_base = col.replace('Home', '')
                    jogo_novo.at[0, col] = get_stat(ultimo_jogo_casa, casa, stat_base)
                elif 'Away' in col:
                    stat_base = col.replace('Away', '')
                    jogo_novo.at[0, col] = get_stat(ultimo_jogo_fora, fora, stat_base)
                elif col == 'H2H_Pontos_Casa':
                    confronto = df[((df['HomeTeam'] == casa) & (df['AwayTeam'] == fora)) | ((df['HomeTeam'] == fora) & (df['AwayTeam'] == casa))]
                    if len(confronto) > 0:
                        ultima_vez = confronto.iloc[-1]
                        if ultima_vez['HomeTeam'] == casa:
                            jogo_novo.at[0, col] = ultima_vez['H2H_Pontos_Casa']
                        else:
                            jogo_novo.at[0, col] = 3.0 - ultima_vez['H2H_Pontos_Casa']
                    else:
                        jogo_novo.at[0, col] = 1.0
            
            jogo_novo = jogo_novo.astype(float)
            
            # --- MODELO 1: 1X2 ---
            probabilidades = modelo_1x2.predict_proba(jogo_novo)[0]
            prob_fora, prob_empate, prob_casa = probabilidades[0], probabilidades[1], probabilidades[2]
            
            # --- MODELO 2: GOLOS (Poisson) ---
            golos_esperados = max(0.1, float(modelo_golos.predict(jogo_novo)[0]))
            p0 = calcular_poisson(golos_esperados, 0)
            p1 = calcular_poisson(golos_esperados, 1)
            p2 = calcular_poisson(golos_esperados, 2)
            p3 = calcular_poisson(golos_esperados, 3)
            p4 = calcular_poisson(golos_esperados, 4)
            
            ov05 = 1 - p0
            ov15 = 1 - (p0 + p1)
            ov25 = 1 - (p0 + p1 + p2)
            ov35 = 1 - (p0 + p1 + p2 + p3)
            ov45 = 1 - (p0 + p1 + p2 + p3 + p4)
            
            # --- MODELO 3: AMBAS MARCAM (BTTS) ---
            prob_btts_lista = modelo_btts.predict_proba(jogo_novo)[0]
            prob_btts_sim = prob_btts_lista[1]
            prob_btts_nao = prob_btts_lista[0]

            # --- APRESENTAÇÃO VISUAL (Tudo em blocos horizontais) ---
            st.divider()
            
            # 1. Vencedor (1X2)
            st.subheader("📊 Mercado Principal (1X2)")
            c1, c2, c3 = st.columns(3)
            c1.metric(label=f"Vitória {casa} (1)", value=f"{prob_casa * 100:.1f}%")
            c2.metric(label="Empate (X)", value=f"{prob_empate * 100:.1f}%")
            c3.metric(label=f"Vitória {fora} (2)", value=f"{prob_fora * 100:.1f}%")
            st.progress(float(max(prob_casa, prob_empate, prob_fora)))
            
            st.write("")
            
            # 2. Dupla Hipótese
            st.subheader("🛡️ Dupla Hipótese")
            dh1, dh2 = st.columns(2)
            dh1.metric(label=f"1X ({casa} ou Empate)", value=f"{(prob_casa + prob_empate) * 100:.1f}%")
            dh2.metric(label=f"X2 ({fora} ou Empate)", value=f"{(prob_fora + prob_empate) * 100:.1f}%")
            
            st.write("")
            
            # 3. Linhas de Golos
            st.subheader(f"🥅 Linhas de Golos Totais (Média Prevista: {golos_esperados:.2f})")
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric(label="+0.5 Golos", value=f"{ov05 * 100:.1f}%")
            g2.metric(label="+1.5 Golos", value=f"{ov15 * 100:.1f}%")
            g3.metric(label="+2.5 Golos", value=f"{ov25 * 100:.1f}%")
            g4.metric(label="+3.5 Golos", value=f"{ov35 * 100:.1f}%")
            g5.metric(label="+4.5 Golos", value=f"{ov45 * 100:.1f}%")

            st.write("")

            # 4. Ambas Marcam (BTTS) em último lugar
            st.subheader("🏁 Ambas Marcam (BTTS)")
            btts1, btts2 = st.columns(2)
            btts1.metric(label="Sim (As duas equipas marcam)", value=f"{prob_btts_sim * 100:.1f}%")
            btts2.metric(label="Não (Pelo menos uma fica a zeros)", value=f"{prob_btts_nao * 100:.1f}%")