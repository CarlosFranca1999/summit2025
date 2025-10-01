# summit2025

# Análise Comparativa de Métodos de Validação Cruzada no Desempenho do Classificador SVM
### *Comparative Analysis of Cross-Validation Methods on the Performance of an SVM Classifier for Heart Disease Prediction*

---

## Visão Geral do Projeto

Este projeto realiza uma análise comparativa de duas estratégias de validação cruzada (CV), **K-Fold** e **StratifiedShuffleSplit**, para avaliar o desempenho de um classificador de Máquina de Vetores de Suporte (SVM) na predição de doenças cardíacas.

A aplicação de modelos de aprendizado de máquina na área biomédica exige avaliações rigorosas para garantir a confiabilidade dos resultados . A validação cruzada é uma técnica fundamental para estimar a capacidade de generalização de um classificador, prevenindo o superajuste (overfitting) . Este estudo explora como a escolha do método de CV pode influenciar a performance estimada do modelo, um fator crucial para sua interpretação e aplicação prática .

## 🎯 Objetivo

O objetivo principal é comparar o desempenho de um classificador SVM na predição de doenças cardíacas, avaliando o impacto de duas estratégias de validação cruzada :
1.  **K-Fold com 10 partições (KFold-10)**
2.  **StratifiedShuffleSplit com 10 divisões e 25% dos dados para teste**

## 📊 Dataset

Foi utilizado o dataset **Heart Disease (Cleveland)**, contendo dados clínicos de 303 pacientes. O conjunto de dados é composto por 13 atributos clínicos (features) e uma variável alvo que indica a presença ou ausência de doença cardíaca .

## 🛠️ Metodologia

A análise foi conduzida em um notebook Jupyter (`HD_Cleveland.ipynb`) seguindo os seguintes passos:

1.  **Pré-processamento:** Os atributos numéricos foram padronizados com a técnica `StandardScaler`. Para evitar vazamento de dados (data leakage), este processo foi integrado a um `Pipeline` do Scikit-learn, garantindo que a padronização fosse aplicada apenas nos dados de treino de cada iteração da validação cruzada .

2.  **Modelo e Otimização:** O modelo de classificação utilizado foi o **Máquina de Vetores de Suporte (SVC)**. Seus hiperparâmetros (`C`, `kernel` e `gamma`) foram otimizados através do `GridSearchCV` para encontrar a melhor combinação para cada estratégia de CV .

3.  **Validação Cruzada:** As duas estratégias (KFold-10 e StratifiedShuffleSplit) foram aplicadas para treinar e validar o modelo . O desempenho foi medido pela acurácia média obtida em cada abordagem .

4.  **Avaliação de Desempenho:** Além da acurácia, foram geradas matrizes de confusão e relatórios de classificação para analisar a distribuição dos erros e a performance do modelo em métricas como precisão, recall e F1-score .

## 📈 Resultados

A análise revelou os seguintes resultados principais:

* **Acurácia Média:** Ambas as estratégias de validação cruzada apresentaram resultados de acurácia média muito próximos .
    * **KFold-10:** Acurácia de **84,16%** .
    * **StratifiedShuffleSplit:** Acurácia de **84,47%** (ligeiramente superior) .

* **Hiperparâmetros Ótimos:** Notou-se que os hiperparâmetros ideais encontrados pelo `GridSearchCV` variaram sutilmente entre as duas abordagens, o que destaca como a estratégia de amostragem pode influenciar a otimização do modelo .

* **Análise de Erros:** A matriz de confusão demonstrou a capacidade do classificador em distinguir entre pacientes saudáveis e doentes, embora com uma pequena quantidade de falsos positivos e falsos negativos .

* **Visualizações:** O notebook contém diversas visualizações para uma análise aprofundada:
    * **Gráficos Comparativos:** Comparam a performance dos modelos com kernel linear e RBF para cada estratégia de CV.
    * **Visualização das Divisões:** Ilustra graficamente como KFold e StratifiedShuffleSplit particionam os dados.
    * **Heatmap de Sensibilidade:** Analisa a interação entre os hiperparâmetros `C` e `gamma` para o kernel RBF.
    * **Visualização com PCA:** Reduz a dimensionalidade dos dados para visualizar a fronteira de decisão do modelo em um gráfico 2D.

## 🏁 Conclusão

Os resultados indicam que, para este conjunto de dados, a escolha entre KFold-10 e StratifiedShuffleSplit não resultou em uma diferença expressiva na estimativa de acurácia .

No entanto, o **StratifiedShuffleSplit** oferece a vantagem de garantir a proporção das classes (estratificação) em cada divisão de treino e teste. Esta é uma prática recomendada para dados biomédicos, onde o desequilíbrio entre classes é comum .

Conclui-se que, embora ambos os métodos sejam eficazes, a escolha da estratégia de validação deve ser consciente e justificada, com a **estratificação sendo um fator importante para a robustez da avaliação do modelo** .

## 🚀 Como Executar o Projeto

Para executar a análise localmente, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone <URL-do-repositorio>
    cd <nome-do-repositorio>
    ```

2.  **Instale as dependências:**
    É recomendado criar um ambiente virtual.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Execute o Notebook:**
    Abra o arquivo `HD_Cleveland.ipynb` em um ambiente Jupyter (como Jupyter Notebook, JupyterLab ou Google Colab) e execute as células sequencialmente. O dataset `Heart_disease_cleveland_new.csv` deve estar no mesmo diretório.

## 📁 Estrutura dos Arquivos

* `HD_Cleveland.ipynb`: Notebook Jupyter contendo todo o código da análise, desde o carregamento dos dados até a visualização dos resultados.
* `Heart_disease_cleveland_new.csv`: O conjunto de dados utilizado no estudo.
* `README.md`: Este arquivo.
