import streamlit as st
import pandas as pd
import joblib

# 1. Configuração da Página
st.set_page_config(page_title="AI Football Predictor", page_icon="⚽", layout="centered")

# 2. Carregar o Cérebro e os Dados
@st.cache_data
def carregar_dados():
    modelo = joblib.load('modelo_xgboost.pkl')
    colunas = joblib.load('colunas.pkl')
    df = pd.read_csv('data/base_processada.csv')
    
    # Descobrir todas as equipas que existem no ficheiro
    equipas = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
    return modelo, colunas, df, equipas

modelo, colunas, df, equipas = carregar_dados()

# 3. Interface Visual (Front-End)
st.title("⚽ Primeira Liga AI Predictor")
st.markdown("Previsões Quantitativas baseadas em **Sistema Elo, Form e Expected Goals (xG)**.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    casa = st.selectbox("🏟️ Equipa da Casa", equipas, index=equipas.index('Porto') if 'Porto' in equipas else 0)
with col2:
    fora = st.selectbox("✈️ Equipa de Fora", equipas, index=equipas.index('Benfica') if 'Benfica' in equipas else 1)

st.write("") # Espaço em branco

if st.button("Gerar Previsão 🔮", use_container_width=True):
    if casa == fora:
        st.error("⚠️ As equipas da Casa e de Fora têm de ser diferentes!")
    else:
        with st.spinner('A analisar o histórico de dados...'):
            # 4. Lógica de Extração: Ir buscar as estatísticas MAIS RECENTES das duas equipas
            stats_casa = df[df['HomeTeam'] == casa].iloc[-1]
            stats_fora = df[df['AwayTeam'] == fora].iloc[-1]
            
            # 5. Criar o "Jogo Novo" no formato exato que o modelo exige
            jogo_novo = pd.DataFrame(columns=colunas)
            jogo_novo.loc[0] = 0.0 
            
            for col in colunas:
                if 'Home' in col:
                    jogo_novo.at[0, col] = stats_casa[col]
                elif 'Away' in col:
                    jogo_novo.at[0, col] = stats_fora[col]
                elif 'H2H' in col:
                    # Procurar o histórico de confrontos diretos entre estas duas equipas
                    confronto = df[((df['HomeTeam'] == casa) & (df['AwayTeam'] == fora)) | ((df['HomeTeam'] == fora) & (df['AwayTeam'] == casa))]
                    if len(confronto) > 0:
                        jogo_novo.at[0, col] = confronto.iloc[-1]['H2H_Pontos_Casa']
                    else:
                        jogo_novo.at[0, col] = 1.0 # Neutro se nunca jogaram
            
            # 6. O Momento da Verdade: Previsão!
            probabilidades = modelo.predict_proba(jogo_novo)[0]
            prob_fora, prob_empate, prob_casa = probabilidades[0], probabilidades[1], probabilidades[2]
            
            # 7. Apresentar os Resultados de forma bonita
            st.divider()
            st.subheader("📊 Probabilidades Matemáticas")
            
            c1, c2, c3 = st.columns(3)
            c1.metric(label=f"Vitória {casa}", value=f"{prob_casa * 100:.1f}%")
            c2.metric(label="Empate", value=f"{prob_empate * 100:.1f}%")
            c3.metric(label=f"Vitória {fora}", value=f"{prob_fora * 100:.1f}%")
            
            # Mostrar uma barra de progresso visual para o favorito
            st.progress(float(max(prob_casa, prob_empate, prob_fora)))