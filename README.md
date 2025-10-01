# Summit 2025 - Análise Comparativa de Métodos de Validação Cruzada

## Análise Comparativa de Métodos de Validação Cruzada no Desempenho do Classificador SVM para Predição de Doenças Cardíacas

Este projeto implementa uma análise comparativa abrangente de diferentes métodos de validação cruzada aplicados a um classificador SVM (Support Vector Machine) para predição de doenças.

## 📋 Descrição

O projeto compara três principais métodos de validação cruzada:
- **K-Fold Cross-Validation**: Divide os dados em k partições de tamanho similar
- **Stratified K-Fold Cross-Validation**: Mantém a proporção de classes em cada partição
- **Leave-One-Out Cross-Validation**: Cada amostra é usada uma vez como validação

### Métricas Avaliadas

Para cada método, avaliamos:
- **Acurácia**: Proporção de predições corretas
- **Precisão**: Proporção de verdadeiros positivos entre predições positivas
- **Recall**: Proporção de verdadeiros positivos identificados
- **F1-Score**: Média harmônica entre precisão e recall

## 🚀 Instalação

### Pré-requisitos

- Python 3.7 ou superior
- pip (gerenciador de pacotes do Python)

### Passos de Instalação

1. Clone o repositório:
```bash
git clone https://github.com/CarlosFranca1999/summit2025.git
cd summit2025
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📊 Uso

### Executar Análise via Script Python

Execute o script principal para realizar a análise completa:

```bash
python svm_cv_analysis.py
```

Este script irá:
1. Carregar e preparar os dados
2. Executar todos os métodos de validação cruzada
3. Gerar visualizações comparativas (salvas como `cv_comparison.png`)
4. Exibir um relatório detalhado no terminal

### Executar Análise via Jupyter Notebook

Para uma análise interativa, use o notebook:

```bash
jupyter notebook analise_svm_cv.ipynb
```

O notebook oferece:
- Explicações detalhadas de cada etapa
- Visualizações interativas
- Possibilidade de modificar parâmetros e experimentar

### Uso Programático

Você também pode usar a classe `SVMCrossValidationAnalysis` em seus próprios scripts:

```python
from svm_cv_analysis import SVMCrossValidationAnalysis

# Inicializa análise
analysis = SVMCrossValidationAnalysis(random_state=42)

# Carrega dados
analysis.load_data()

# Executa análise com k=5 e k=10
analysis.run_all_methods(k_values=[5, 10], include_loo=False)

# Gera visualizações
analysis.create_comparison_plots('resultados.png')

# Gera relatório
analysis.generate_report()
```

## 📁 Estrutura do Projeto

```
summit2025/
├── README.md                    # Documentação do projeto
├── requirements.txt             # Dependências do projeto
├── .gitignore                   # Arquivos ignorados pelo Git
├── svm_cv_analysis.py          # Script principal de análise
├── analise_svm_cv.ipynb        # Notebook Jupyter interativo
└── cv_comparison.png           # Gráficos de comparação (gerado)
```

## 🔬 Metodologia

### Dataset

O projeto utiliza o dataset de câncer de mama do scikit-learn como proxy para análise de doenças. Este dataset contém:
- 569 amostras
- 30 características numéricas
- 2 classes (benigno/maligno)

### Classificador SVM

Parâmetros utilizados:
- Kernel: RBF (Radial Basis Function)
- C: 1.0
- Gamma: 'scale'

### Validação Cruzada

#### K-Fold (k=5 e k=10)
Divide o dataset em k partições de tamanho aproximadamente igual. Em cada iteração, k-1 partições são usadas para treinamento e 1 para validação.

**Vantagens:**
- Uso eficiente dos dados
- Balanceamento entre viés e variância
- Computacionalmente eficiente

**Desvantagens:**
- Pode não preservar a proporção de classes

#### Stratified K-Fold (k=5 e k=10)
Similar ao K-Fold, mas garante que cada partição mantenha a mesma proporção de classes do dataset original.

**Vantagens:**
- Preserva a distribuição de classes
- Ideal para datasets desbalanceados
- Estimativas mais confiáveis

**Desvantagens:**
- Ligeiramente mais complexo

#### Leave-One-Out (LOO)
Cada amostra é usada uma vez como validação, enquanto todas as outras são usadas para treinamento.

**Vantagens:**
- Uso máximo dos dados
- Estimativa menos enviesada

**Desvantagens:**
- Computacionalmente custoso
- Alta variância
- Não implementado por padrão devido ao custo

## 📈 Resultados Esperados

O script gera:

1. **Gráficos comparativos** mostrando:
   - Comparação de acurácia entre métodos
   - Comparação de múltiplas métricas
   - Distribuição de acurácia (box plots)
   - Tabela resumo dos resultados

2. **Relatório textual** contendo:
   - Estatísticas descritivas do dataset
   - Resultados detalhados por método
   - Análise comparativa
   - Identificação do melhor método

## 🔧 Personalização

### Modificar valores de k

```python
analysis.run_all_methods(k_values=[3, 5, 7, 10])
```

### Incluir Leave-One-Out

```python
analysis.run_all_methods(k_values=[5, 10], include_loo=True)
```

### Usar seu próprio dataset

```python
# Substitua o método load_data() com seus dados
analysis.X = seu_X
analysis.y = seu_y
```

### Modificar parâmetros do SVM

Edite o método `create_svm_classifier()` em `svm_cv_analysis.py`:

```python
def create_svm_classifier(self):
    return SVC(kernel='linear', C=10.0, gamma='auto', random_state=self.random_state)
```

## 📚 Dependências

- **numpy**: Computação numérica
- **pandas**: Manipulação de dados
- **scikit-learn**: Machine learning e validação cruzada
- **matplotlib**: Visualização de dados
- **seaborn**: Visualização estatística
- **jupyter**: Notebooks interativos

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir novas funcionalidades
- Enviar pull requests

## 📝 Licença

Este projeto é fornecido "como está" para fins educacionais e de pesquisa.

## 👥 Autor

Carlos França - [CarlosFranca1999](https://github.com/CarlosFranca1999)

## 📧 Contato

Para questões ou sugestões, abra uma issue no GitHub.

## 🔍 Referências

- Scikit-learn Documentation: https://scikit-learn.org/
- Cross-validation (statistics): https://en.wikipedia.org/wiki/Cross-validation_(statistics)
- Support Vector Machines: https://en.wikipedia.org/wiki/Support_vector_machine

---

**Nota**: Este projeto foi desenvolvido como parte do Summit 2025 para demonstrar técnicas de validação cruzada em classificação de doenças usando SVM.