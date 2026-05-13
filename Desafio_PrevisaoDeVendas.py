import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.header("Previsão de Vendas")

# Dados: [Investimento em Marketing] -> Faturamento
dados_vendas = pd.DataFrame({
    'investimento': [100, 200, 300, 400, 500, 600],
    'faturamento': [1200, 2500, 3200, 4800, 5100, 6300]
})

# objetivo: previsão de FATURAMENTO 

st.scatter_chart(dados_vendas, x='investimento', y='faturamento')
modelo_vendas = LinearRegression()
modelo_vendas.fit(dados_vendas[['investimento']], dados_vendas['faturamento'])
investimento_input = st.slider('Investimento em Marketing', 0, 1000, 300)
faturamento_previsto = modelo_vendas.predict([[investimento_input]])
st.metric('Faturamento Previsto', f'R$ {faturamento_previsto[0]:.1f}')
