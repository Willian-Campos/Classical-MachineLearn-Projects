# Análise de Risco de Acidentes Offshore
### *Projeto de Engenharia e Ciência de Dados*

Este repositório apresenta um estudo completo sobre **riscos de acidentes em ambiente offshore**, incluindo **limpeza, modelagem, enriquecimento financeiro e análise estatística**.  
O projeto combina técnicas de *data engineering*, *data analysis* e *visual analytics* para identificar padrões relevantes que podem apoiar decisões de segurança e redução de custos.

---

## Objetivo do Projeto
Desenvolver uma análise estruturada dos incidentes e quase-incidentes offshore, identificando:  
- Tendências temporais e operacionais  
- Distribuição por severidade  
- Padrões comportamentais  
- Impactos financeiros  
- Pontos críticos para mitigação de risco  

> **"Prevenção é melhor que remediação."**

---

## Estrutura do Projeto

### **1. `risco_acidente_offshore.ipynb`**  
Notebook dedicado à **limpeza, padronização e engenharia de dados**, incluindo:  
- Tratamento de inconsistências  
- Normalização de colunas  
- Criação de variáveis categóricas e temporais  
- Preparação para análises estatísticas e financeiras

---

### **2. `risco_acidente_offshore_FINANCEIRO.ipynb`**  
Notebook focado no **impacto monetário** dos acidentes.  
Principais operações:  
- Pesquisa de custos típicos por severidade  
- Criação de colunas de impacto econômico  
- Consolidação de custos totais para cada categoria  
- Estimativas baseadas em literatura e dados oficiais (quando disponíveis)

---

### **3. `risco_acidente_offshore_ANALISE.ipynb`**  
Notebook destinado à **análise estatística e visual**, incluindo:  
- Distribuição de incidentes por horário  
- Análise por tipo de instalação  
- Frequência por categoria (leve, moderado, grave)  
- Discussões sobre causas mais prováveis  
- Visualizações para apoio à tomada de decisão  

---

## 📈 Principais Insights
- Maior incidência de incidentes próximos aos horários de **troca de turno**  
- Certas instalações concentram mais ocorrências  
- Incidentes graves são raros, mas apresentam **impacto financeiro desproporcionalmente alto**  
- O custo agregado total mostra prejuízos bilionários, reforçando a importância de processos preventivos

---

## 🛠️ Tecnologias Utilizadas
- Python (Jupyter Notebook)  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Feather / CSV  

---

## 📦 Estrutura de Arquivos
