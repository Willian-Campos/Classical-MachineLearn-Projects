import streamlit as st
import pandas as pd
import joblib
from pycaret.classification import load_model, predict_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.metrics import confusion_matrix

# Carregar o modelo salvo
model = load_model('tuned_rf_model')

# Definir as listas para cada tipo de variável
qualitativas = ['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos', 'tipo_renda', 'educacao',
               'estado_civil', 'tipo_residencia', 'qt_pessoas_residencia']
quantitativas = ['idade', 'tempo_emprego', 'renda']

# Funções de pré-processamento
def remover_colunas(df):
    return df.drop(columns=['index', 'data_ref'])

# ESSA PARTE SERVE APENAS PARA TREINAR O MODELO - deixado aqui para não esquecer que ela existe no notebook .ipynb
#def balanceamento(df):
#    df_1 = df[df['mau'] == 1]
#    n = df_1.shape[0] * 3
#    df_0 = df[df['mau'] == 0].sample(n=n, random_state=0)
#    return pd.concat([df_1, df_0]).sample(frac=1, random_state=0).reset_index(drop=True)

def preencher_missings(df):
    df['tempo_emprego'].fillna(df['tempo_emprego'].mean(), inplace=True)
    return df

def remove_outliers_renda(df, z_thresh=3):
    col = 'renda'
    col_zscore = (df[col] - df[col].mean()) / df[col].std()
    df = df[(col_zscore.abs() <= z_thresh)]
    return df

def agrupar_categorias(df):
    df['qt_pessoas_residencia'] = df['qt_pessoas_residencia'].astype(int)
    df['qt_pessoas_residencia'].replace({6: '6+', 9: '6+', 15: '6+', 7: '6+'}, inplace=True)
    df['qtd_filhos'].replace({5: '5+', 7: '5+', 14: '5+'}, inplace=True)
    df['tipo_renda'].replace({'Bolsista': 'Assalariado'}, inplace=True)
    df['educacao'].replace({'Fundamental': 'Básico', 'Médio': 'Básico', 
                            'Superior incompleto': 'Avançado', 
                            'Superior completo': 'Avançado',
                            'Pós graduação': 'Avançado'}, inplace=True)
    df['tipo_residencia'].replace({'Estúdio': 'Outros', 'Comunitário': 'Outros', 
                                   'Governamental': 'Outros'}, inplace=True)
    return df

def one_hot_encoding(df):
    df = pd.get_dummies(df, columns=qualitativas, drop_first=True)
    return df

def min_max_scaler(df):
    scaler = MinMaxScaler()
    df[quantitativas] = scaler.fit_transform(df[quantitativas])
    return df

pipeline = Pipeline(steps=[
    ('remover_colunas', FunctionTransformer(remover_colunas)), 
    #('balanceamento', FunctionTransformer(balanceamento)), 
    ('preencher_missings', FunctionTransformer(preencher_missings)),
    ('remove_outliers_renda', FunctionTransformer(remove_outliers_renda)),
    ('agrupar_categorias', FunctionTransformer(agrupar_categorias)),
    ('one_hot_encoding', FunctionTransformer(one_hot_encoding)),
    ('min_max_scaler', FunctionTransformer(min_max_scaler)),
])

# Função para fazer previsões
def fazer_previsao(dados):
    dados_preprocessados = pipeline.transform(dados)
    previsoes = predict_model(model, data=dados_preprocessados)
    return previsoes

# Função para calcular lucro
def calcular_lucro_total(cm, ganho_TN, perda_FP, perda_FN):
    # Extraindo TN, FP, FN da matriz de confusão
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    # Calcular lucro total
    lucro = (TN * ganho_TN) - (FN * perda_FN)
    lucro_perdido = FP * perda_FP
    return lucro, lucro_perdido

# Configurar a interface do Streamlit
st.set_page_config(page_title="CreditScore")

st.title("Aplicativo de Previsão de Crédito")
st.write("Carregue um arquivo para fazer previsões")

# Campos de entrada para ganho e perda
ganho_TN = st.number_input("Ganho por True Negative (Adimplentes)", value=1, step=1)
perda_FP = st.number_input("Perda por False Positive (Adimplentes classificados como inadimplentes)", value=1, step=1)
perda_FN = st.number_input("Perda por False Negative (Inadimplentes classificados como adimplentes)", value=10, step=1)

# Carregar arquivo
arquivo_carregado = st.file_uploader("Escolha um arquivo", type=["csv", "xlsx", "ftr"])

if arquivo_carregado is not None:
    try:
        # Identificar o tipo de arquivo e carregar os dados
        if arquivo_carregado.name.endswith('.csv'):
            dados = pd.read_csv(arquivo_carregado)
        elif arquivo_carregado.name.endswith('.xlsx'):
            dados = pd.read_excel(arquivo_carregado)
        elif arquivo_carregado.name.endswith('.ftr'):
            dados = pd.read_feather(arquivo_carregado)
        else:
            st.error("Formato de arquivo não suportado!")
        
        # Mostrar os dados carregados
        st.write("Dados carregados:")
        st.write(dados)

        # Fazer previsões
        previsoes = fazer_previsao(dados)

        # Mostrar as previsões
        st.write("Previsões:")
        st.write(previsoes)

        # Extraindo valores reais e previstos
        y_true = previsoes['mau']  # Alvo real
        y_pred = previsoes['prediction_label']  # Predições feitas pelo modelo

        # Calcular matriz de confusão
        cm = confusion_matrix(y_true, y_pred)

        # Calcular lucro total e lucro perdido
        lucro, lucro_perdido = calcular_lucro_total(cm, ganho_TN, perda_FP, perda_FN)

        # Mostrar lucro total e lucro perdido
        st.write(f"Lucro total: {lucro}")
        st.write(f"Lucro perdido (devido a FalsePositive): {lucro_perdido}")

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

# Rodapé com link para o GitHub
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp { bottom: 50px; }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            text-align: center;
            padding: 10px;
            font-size: small;
        }
    </style>
    <div class="footer">
        <p>Para mais informações sobre o modelo, visite o <a href="https://github.com/Willian-Campos/DataScienceProjects/tree/master/ebac_ciencia-de-dados_projeto-final_streamlit-e-pipeline" target="_blank">repositório no GitHub</a>.</p>
    </div>
    """, unsafe_allow_html=True)