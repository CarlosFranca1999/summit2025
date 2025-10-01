# Quick Start Guide

## 🚀 Início Rápido

### Instalação em 3 Passos

```bash
# 1. Clone o repositório
git clone https://github.com/CarlosFranca1999/summit2025.git
cd summit2025

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a análise
python svm_cv_analysis.py
```

### Resultado Esperado

O script irá:
- ✅ Carregar 569 amostras com 30 características
- ✅ Executar 4 métodos de validação cruzada
- ✅ Gerar visualizações em `cv_comparison.png`
- ✅ Exibir relatório completo no terminal

### Exemplo de Saída

```
================================================================================
ANÁLISE COMPARATIVA DE MÉTODOS DE VALIDAÇÃO CRUZADA
Classificador SVM para Predição de Doenças Cardíacas
================================================================================

=== K-Fold Cross-Validation (k=5) ===
Accuracy:  0.9789 (+/- 0.0071)
Precision: 0.9775 (+/- 0.0145)
Recall:    0.9892 (+/- 0.0098)
F1-Score:  0.9832 (+/- 0.0058)

...

Melhor método (Acurácia): K-Fold
  Acurácia: 0.9789 (+/- 0.0071)
```

## 📊 Uso Interativo

### Jupyter Notebook

```bash
jupyter notebook analise_svm_cv.ipynb
```

Execute célula por célula para:
- Ver explicações detalhadas
- Modificar parâmetros interativamente
- Gerar visualizações customizadas

## 🔧 Uso Programático

### Exemplo Básico

```python
from svm_cv_analysis import SVMCrossValidationAnalysis

# Inicializa
analysis = SVMCrossValidationAnalysis(random_state=42)

# Carrega dados
analysis.load_data()

# Executa análise
results = analysis.run_all_methods(k_values=[5, 10])

# Gera visualizações
analysis.create_comparison_plots('meus_resultados.png')

# Gera relatório
analysis.generate_report()
```

### Exemplo Avançado

```python
# Testar diferentes valores de k
analysis.run_all_methods(k_values=[3, 5, 7, 10])

# Incluir Leave-One-Out (cuidado: pode ser lento!)
analysis.run_all_methods(k_values=[5], include_loo=True)

# Executar método específico
result = analysis.k_fold_cv(n_splits=7)
print(f"Acurácia: {result['accuracy_mean']:.4f}")
```

## 📦 Estrutura de Arquivos

```
summit2025/
├── README.md              📖 Documentação completa
├── QUICKSTART.md          🚀 Este guia rápido
├── SUMMARY.md             📋 Resumo da implementação
├── requirements.txt       📦 Dependências
├── svm_cv_analysis.py     🐍 Script principal
├── analise_svm_cv.ipynb   📓 Notebook interativo
└── cv_comparison.png      📊 Visualizações (gerado)
```

## 🎯 Casos de Uso

### 1. Análise Rápida
**Objetivo**: Ver resultados imediatamente
```bash
python svm_cv_analysis.py
```

### 2. Exploração Interativa
**Objetivo**: Aprender e experimentar
```bash
jupyter notebook analise_svm_cv.ipynb
```

### 3. Integração em Projeto
**Objetivo**: Usar em seu próprio código
```python
from svm_cv_analysis import SVMCrossValidationAnalysis
# Seu código aqui
```

## 🔍 Personalização Rápida

### Mudar Número de Folds

```python
# Testar k=3 e k=7
analysis.run_all_methods(k_values=[3, 7])
```

### Usar Seus Próprios Dados

```python
analysis = SVMCrossValidationAnalysis()
analysis.X = seu_array_X  # numpy array
analysis.y = seu_array_y  # numpy array
results = analysis.run_all_methods()
```

### Modificar Parâmetros do SVM

Edite em `svm_cv_analysis.py`:
```python
def create_svm_classifier(self):
    return SVC(
        kernel='rbf',      # 'linear', 'poly', 'rbf', 'sigmoid'
        C=1.0,             # Parâmetro de regularização
        gamma='scale',     # 'scale', 'auto', ou valor numérico
        random_state=self.random_state
    )
```

## ⚡ Performance

| Método | Tempo Estimado | Acurácia Típica |
|--------|---------------|-----------------|
| K-Fold (k=5) | ~5 segundos | 97-98% |
| K-Fold (k=10) | ~10 segundos | 97-98% |
| Stratified K-Fold (k=5) | ~5 segundos | 97-98% |
| Stratified K-Fold (k=10) | ~10 segundos | 97-98% |
| Leave-One-Out | ~5 minutos | 97-98% |

*Tempos baseados em dataset de 569 amostras*

## 🐛 Troubleshooting

### Erro: Module not found
```bash
pip install -r requirements.txt
```

### Erro: Permissão negada
```bash
chmod +x svm_cv_analysis.py
python svm_cv_analysis.py
```

### Gráficos não aparecem no Jupyter
```python
%matplotlib inline
```

### Out of memory (LOO)
```python
# Não use LOO com datasets grandes
analysis.run_all_methods(k_values=[5, 10], include_loo=False)
```

## 📚 Próximos Passos

1. **Leia o README.md** para documentação completa
2. **Explore o SUMMARY.md** para entender a implementação
3. **Execute o notebook** para análise interativa
4. **Customize o código** para suas necessidades

## 💡 Dicas

- Use `k=5` para análise rápida
- Use `k=10` para melhor estimativa
- Use Stratified K-Fold para dados desbalanceados
- Evite LOO para datasets grandes (>1000 amostras)
- Fixe `random_state` para reprodutibilidade

## 📞 Suporte

- **Documentação**: Veja README.md
- **Exemplos**: Veja analise_svm_cv.ipynb
- **Issues**: Abra um issue no GitHub

---

**Tempo para começar**: < 5 minutos  
**Nível de dificuldade**: Iniciante  
**Linguagem**: Python 3.7+
