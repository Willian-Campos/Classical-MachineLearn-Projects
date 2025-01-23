# Classical Machine Learn Projects

Este repositório contém uma coleção dos meus principais projetos de **Ciência de Dados**, abrangendo uma variedade de tópicos e desafios, desde **análises estatísticas** até **modelos complexos** como **redes neurais**. O objetivo é demonstrar minha experiência e habilidades na aplicação de técnicas de análise de dados e aprendizado de máquina para resolver problemas reais.

### Tópicos abordados:

- **Análise Estatística**  
  Projetos envolvendo estatísticas descritivas, inferenciais e outras análises exploratórias de dados.

- **Árvores de Decisão e Florestas Aleatórias**  
  Modelos para classificação e regressão baseados em árvores de decisão.

- **Machine Learning Supervisionado e Não Supervisionado**  
  Técnicas como regressão, classificação, clustering (K-Means, DBSCAN) e PCA (Análise de Componentes Principais).

- **Redes Neurais**  
  Desenvolvimento e treinamento de modelos de redes neurais profundas para tarefas de classificação, regressão e previsão.

- **Análises Temporais e Séries Temporais**  
  Análises de dados sequenciais com foco em previsão e modelos de séries temporais.

- **Automatização de Processos e Pipelines**  
  Automatização de processos de ETL e criação de pipelines de dados para facilitar o fluxo de trabalho.

### Estrutura do repositório

Cada projeto está organizado dentro de seu próprio diretório com o seguinte formato:

- **notebooks**: Códigos em Jupyter Notebooks para análise interativa.
- **scripts**: Scripts Python utilizados para automação e execução de tarefas.
- **datasets**: Conjunto de dados utilizados no projeto, com links de download ou arquivos diretamente no repositório.
- **outputs**: Resultados gerados, incluindo gráficos, relatórios e modelos treinados.
- **readme.md**: Descrição do projeto, objetivos, técnicas usadas e instruções de execução.

### Projetos Principais

1. **[Análise Temporal da Bolsa de Valores](https://github.com/Willian-Campos/DataScienceProjects/tree/master/projeto_mercado_financeiro_metaTrader)**
   - **Descrição**: Este projeto utiliza a biblioteca **MetaTrader5** para acessar dados de mercado em tempo real e alta frequência, com mais de 20 milhões de registros. A análise enfoca o desafio de dados desbalanceados e a importância de uma visão analítica crítica ao trabalhar com dados financeiros em alta frequência.

2. **[Clusterização de Clientes](https://github.com/Willian-Campos/DataScienceProjects/tree/master/ebac_ciencia-de-dados_projeto03_clustering)**
   - **Descrição**: Projeto que utiliza técnicas de **K-Means** e a métrica de **Gower** para realizar agrupamentos de clientes com base em seus hábitos de compra online, visando identificar padrões de comportamento e segmentar os consumidores de maneira estratégica.

3. **[Automatização de Processos com Pipelines](https://github.com/Willian-Campos/DataScienceProjects/tree/master/ebac_ciencia-de-dados_projeto-final_streamlit-e-pipeline)**
   - **Descrição**: Este projeto visa automatizar o fluxo de trabalho de análise de dados utilizando **Pycaret**, **Streamlit** e **pipelines**. A ideia é transformar dados brutos em informações úteis através de um processo automatizado, com especial atenção para a comparação de métricas de modelos de aprendizado de máquina, como **Recall** e **F1Score**.

4. **[Predição de Inadimplência de Clientes](https://github.com/Willian-Campos/DataScienceProjects/tree/master/ebac_ciencia-de-dados_projeto02_regressao-compara-modelos)**
   - **Descrição**: Projeto focado na predição de inadimplência de clientes, utilizando técnicas de **Árvores de Decisão** e **Machine Learning** para prever a probabilidade de um cliente ser inadimplente com base em dados financeiros e de crédito.

5. **[Previsão de Renda com Machine Learning](https://github.com/Willian-Campos/DataScienceProjects/tree/master/ebac_ciencia-de-dados_projeto01_random-forest)**
   - **Descrição**: Modelo preditivo para estimar a renda de indivíduos a partir de variáveis explicativas. O projeto segue as etapas do modelo **CRISP-DM**, desde a exploração de dados até a modelagem com **Árvores de Decisão** e **Random Forest**.

### Como rodar os projetos

1. **Clonar o repositório**:
   - Use o comando `git clone` para obter uma cópia local do repositório:
   
     ```bash
     git clone https://github.com/Willian-Campos/DataScienceProjects.git
     ```

2. **Instalar dependências**:
   - A maioria dos projetos usa bibliotecas populares de Python. Para instalar as dependências necessárias, crie um ambiente virtual e use o arquivo `requirements.txt`:

     ```bash
     python -m venv venv
     source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
     pip install -r requirements.txt
     ```

3. **Rodar os notebooks**:
   - Abra os notebooks Jupyter:

     ```bash
     jupyter notebook
     ```

4. **Verificação dos resultados**:
   - Os resultados podem ser encontrados na pasta **`outputs/`** ou exibidos diretamente nos notebooks.

### Contribuindo

Se você deseja contribuir com este repositório, fique à vontade para enviar **pull requests** com melhorias ou novos projetos. As contribuições são sempre bem-vindas!

### Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

**Contato:**

- [LinkedIn](https://www.linkedin.com/in/willian-augusto-campos/)
- [E-mail](mailto:willian.augusto.campos@gmail.com)
- [GitHub](https://github.com/Willian-Campos)
