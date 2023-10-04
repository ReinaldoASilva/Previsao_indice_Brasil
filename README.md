# README - Previsão de Séries Temporais com PyCaret

Este repositório contém um código em Python para realizar previsões de séries temporais utilizando a biblioteca PyCaret. O objetivo é prever os preços de fechamento, máximo, mínimo e abertura do índice BVSP (Bovespa).

## Pré-requisitos

Antes de executar o código, certifique-se de ter as seguintes bibliotecas instaladas:

- yfinance
- pandas
- pandas_profiling
- plotly_express
- pycaret.regression

Você pode instalá-las usando o gerenciador de pacotes pip:

```
pip install yfinance pandas pandas_profiling plotly_express pycaret
```

## Uso

O código está dividido em várias etapas, que são descritas a seguir:

1. Importação das bibliotecas: As bibliotecas necessárias são importadas no início do código.

2. Busca e pré-processamento dos dados: Os dados históricos do índice BVSP são obtidos usando a biblioteca yfinance. Em seguida, os dados são pré-processados removendo colunas desnecessárias e calculando médias móveis (MM20d e MM200d) para o preço de fechamento.

3. Geração de um relatório de perfil: A biblioteca pandas_profiling é utilizada para gerar um relatório de perfil dos dados do BVSP, fornecendo insights e análises estatísticas do conjunto de dados.

4. Divisão dos dados: Os últimos 365 dias de dados são separados como conjunto de teste (BVSP_prever), e os dados restantes são usados para treinamento (BVSP).

5. Configuração da tarefa de regressão: A função setup do pycaret.regression é chamada para inicializar a tarefa de regressão, especificando os dados, a variável alvo ('Close' no primeiro caso) e um ID de sessão único.

6. Comparação e seleção de modelos: A função compare_models é usada para comparar e selecionar automaticamente os 3 melhores modelos de regressão.

7. Criação e ajuste de modelos: Dois modelos, 'lar' (Least Angle Regression) e 'ridge' (Ridge Regression), são criados usando a função create_model. Em seguida, a função tune_model é usada para ajustar os hiperparâmetros desses modelos usando uma busca em grade e validação cruzada.

8. Visualização do desempenho do modelo: A função plot_model é usada para visualizar as métricas de desempenho e importância das características dos modelos ajustados.

9. Geração de previsões: A função predict_model é usada para gerar previsões no conjunto de teste usando o modelo ajustado.

10. Finalização do modelo: A função finalize_model é chamada para treinar o modelo em todo o conjunto de dados (BVSP) e finalizá-lo para implantação.

11. Visualização dos preços previstos: Os preços previstos (tanto de fechamento quanto máximo) são visualizados usando a biblioteca plotly_express, gerando gráficos de linha comparando os preços reais e previstos.

12. As etapas de 4 a 11 são repetidas para prever os preços mínimos e de abertura, com pequenas modificações na variável alvo e nas colunas de dados.

Certifique-se de ter acesso aos dados históricos do BVSP e ajuste o código conforme necessário para executar a previsão.

## Contribuição

Contribuições são bem-vindas! Se você tiver sugestões de melhorias, correções de bugs ou novos recursos, fique à vontade para abrir uma issue ou enviar um pull request.


 
