"""
Análise Comparativa de Métodos de Validação Cruzada no Desempenho do 
Classificador SVM para Predição de Doenças Cardíacas

Este script implementa uma análise comparativa de diferentes métodos de 
validação cruzada (K-Fold, Stratified K-Fold, Leave-One-Out) aplicados 
a um classificador SVM para predição de doenças cardíacas.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score, 
    KFold, 
    StratifiedKFold, 
    LeaveOneOut,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SVMCrossValidationAnalysis:
    """
    Classe para realizar análise comparativa de métodos de validação cruzada
    em classificadores SVM.
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa a análise.
        
        Parameters:
        -----------
        random_state : int
            Semente para reprodutibilidade
        """
        self.random_state = random_state
        self.results = {}
        self.X = None
        self.y = None
        
    def load_data(self):
        """
        Carrega e prepara os dados para análise.
        Utiliza o dataset de câncer de mama como proxy para doenças cardíacas
        (ambos são problemas de classificação binária em saúde).
        """
        # Carrega dados
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target
        
        # Normalização dos dados
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1]} características")
        print(f"Distribuição de classes: {np.bincount(self.y)}")
        
    def create_svm_classifier(self):
        """
        Cria um classificador SVM com parâmetros otimizados.
        
        Returns:
        --------
        SVC : Classificador SVM
        """
        return SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state)
    
    def k_fold_cv(self, n_splits=10):
        """
        Realiza validação cruzada K-Fold.
        
        Parameters:
        -----------
        n_splits : int
            Número de folds
            
        Returns:
        --------
        dict : Resultados da validação
        """
        print(f"\n=== K-Fold Cross-Validation (k={n_splits}) ===")
        
        clf = self.create_svm_classifier()
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Define scorers
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        
        # Executa validação cruzada
        scores = cross_validate(clf, self.X, self.y, cv=cv, scoring=scoring, 
                               return_train_score=False, n_jobs=-1)
        
        results = {
            'method': 'K-Fold',
            'n_splits': n_splits,
            'accuracy_mean': scores['test_accuracy'].mean(),
            'accuracy_std': scores['test_accuracy'].std(),
            'precision_mean': scores['test_precision'].mean(),
            'precision_std': scores['test_precision'].std(),
            'recall_mean': scores['test_recall'].mean(),
            'recall_std': scores['test_recall'].std(),
            'f1_mean': scores['test_f1'].mean(),
            'f1_std': scores['test_f1'].std(),
            'scores': scores
        }
        
        self._print_results(results)
        return results
    
    def stratified_k_fold_cv(self, n_splits=10):
        """
        Realiza validação cruzada Stratified K-Fold.
        
        Parameters:
        -----------
        n_splits : int
            Número de folds
            
        Returns:
        --------
        dict : Resultados da validação
        """
        print(f"\n=== Stratified K-Fold Cross-Validation (k={n_splits}) ===")
        
        clf = self.create_svm_classifier()
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Define scorers
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }
        
        # Executa validação cruzada
        scores = cross_validate(clf, self.X, self.y, cv=cv, scoring=scoring, 
                               return_train_score=False, n_jobs=-1)
        
        results = {
            'method': 'Stratified K-Fold',
            'n_splits': n_splits,
            'accuracy_mean': scores['test_accuracy'].mean(),
            'accuracy_std': scores['test_accuracy'].std(),
            'precision_mean': scores['test_precision'].mean(),
            'precision_std': scores['test_precision'].std(),
            'recall_mean': scores['test_recall'].mean(),
            'recall_std': scores['test_recall'].std(),
            'f1_mean': scores['test_f1'].mean(),
            'f1_std': scores['test_f1'].std(),
            'scores': scores
        }
        
        self._print_results(results)
        return results
    
    def leave_one_out_cv(self):
        """
        Realiza validação cruzada Leave-One-Out.
        
        Returns:
        --------
        dict : Resultados da validação
        """
        print(f"\n=== Leave-One-Out Cross-Validation ===")
        print("Atenção: Este método pode ser computacionalmente intensivo...")
        
        clf = self.create_svm_classifier()
        cv = LeaveOneOut()
        
        # Para LOO, calculamos apenas acurácia devido ao custo computacional
        accuracy_scores = cross_val_score(clf, self.X, self.y, cv=cv, 
                                         scoring='accuracy', n_jobs=-1)
        
        results = {
            'method': 'Leave-One-Out',
            'n_splits': len(self.y),
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'precision_mean': None,
            'precision_std': None,
            'recall_mean': None,
            'recall_std': None,
            'f1_mean': None,
            'f1_std': None,
            'scores': {'test_accuracy': accuracy_scores}
        }
        
        print(f"Accuracy: {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")
        return results
    
    def _print_results(self, results):
        """
        Imprime resultados da validação cruzada.
        
        Parameters:
        -----------
        results : dict
            Dicionário com resultados
        """
        print(f"Accuracy:  {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")
        print(f"Precision: {results['precision_mean']:.4f} (+/- {results['precision_std']:.4f})")
        print(f"Recall:    {results['recall_mean']:.4f} (+/- {results['recall_std']:.4f})")
        print(f"F1-Score:  {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")
    
    def run_all_methods(self, k_values=[5, 10], include_loo=False):
        """
        Executa todos os métodos de validação cruzada.
        
        Parameters:
        -----------
        k_values : list
            Lista com valores de k para testar
        include_loo : bool
            Se True, inclui Leave-One-Out (pode ser lento)
            
        Returns:
        --------
        dict : Todos os resultados
        """
        self.results = {}
        
        # K-Fold
        for k in k_values:
            key = f'kfold_{k}'
            self.results[key] = self.k_fold_cv(n_splits=k)
        
        # Stratified K-Fold
        for k in k_values:
            key = f'stratified_kfold_{k}'
            self.results[key] = self.stratified_k_fold_cv(n_splits=k)
        
        # Leave-One-Out (opcional)
        if include_loo:
            self.results['loo'] = self.leave_one_out_cv()
        
        return self.results
    
    def create_comparison_plots(self, save_path='cv_comparison.png'):
        """
        Cria gráficos comparativos dos métodos de validação cruzada.
        
        Parameters:
        -----------
        save_path : str
            Caminho para salvar a figura
        """
        if not self.results:
            print("Execute run_all_methods() primeiro!")
            return
        
        # Prepara dados para visualização
        methods = []
        accuracy_means = []
        accuracy_stds = []
        precision_means = []
        recall_means = []
        f1_means = []
        
        for key, result in self.results.items():
            methods.append(result['method'] + f" (k={result['n_splits']})" 
                          if result['n_splits'] < 100 else result['method'])
            accuracy_means.append(result['accuracy_mean'])
            accuracy_stds.append(result['accuracy_std'])
            
            if result['precision_mean'] is not None:
                precision_means.append(result['precision_mean'])
                recall_means.append(result['recall_mean'])
                f1_means.append(result['f1_mean'])
            else:
                precision_means.append(0)
                recall_means.append(0)
                f1_means.append(0)
        
        # Cria figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise Comparativa de Métodos de Validação Cruzada\n' +
                     'Classificador SVM para Predição de Doenças', 
                     fontsize=16, fontweight='bold')
        
        # 1. Gráfico de barras - Acurácia
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(methods)), accuracy_means, yerr=accuracy_stds, 
                       capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Método de Validação Cruzada', fontweight='bold')
        ax1.set_ylabel('Acurácia', fontweight='bold')
        ax1.set_title('Comparação de Acurácia entre Métodos', fontweight='bold')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0.85, 1.0])
        
        # Adiciona valores nas barras
        for i, (bar, val) in enumerate(zip(bars, accuracy_means)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Gráfico de barras - Todas as métricas
        ax2 = axes[0, 1]
        x = np.arange(len(methods))
        width = 0.2
        
        # Filtra resultados que têm todas as métricas
        valid_indices = [i for i, p in enumerate(precision_means) if p > 0]
        valid_methods = [methods[i] for i in valid_indices]
        valid_accuracy = [accuracy_means[i] for i in valid_indices]
        valid_precision = [precision_means[i] for i in valid_indices]
        valid_recall = [recall_means[i] for i in valid_indices]
        valid_f1 = [f1_means[i] for i in valid_indices]
        
        if valid_methods:
            x_valid = np.arange(len(valid_methods))
            ax2.bar(x_valid - 1.5*width, valid_accuracy, width, label='Accuracy', 
                   color='skyblue', edgecolor='black', alpha=0.7)
            ax2.bar(x_valid - 0.5*width, valid_precision, width, label='Precision', 
                   color='lightgreen', edgecolor='black', alpha=0.7)
            ax2.bar(x_valid + 0.5*width, valid_recall, width, label='Recall', 
                   color='lightcoral', edgecolor='black', alpha=0.7)
            ax2.bar(x_valid + 1.5*width, valid_f1, width, label='F1-Score', 
                   color='plum', edgecolor='black', alpha=0.7)
            
            ax2.set_xlabel('Método de Validação Cruzada', fontweight='bold')
            ax2.set_ylabel('Score', fontweight='bold')
            ax2.set_title('Comparação de Múltiplas Métricas', fontweight='bold')
            ax2.set_xticks(x_valid)
            ax2.set_xticklabels(valid_methods, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim([0.85, 1.0])
        
        # 3. Box plot - Distribuição de acurácia
        ax3 = axes[1, 0]
        accuracy_data = [result['scores']['test_accuracy'] 
                        for result in self.results.values()]
        bp = ax3.boxplot(accuracy_data, labels=methods, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Método de Validação Cruzada', fontweight='bold')
        ax3.set_ylabel('Acurácia', fontweight='bold')
        ax3.set_title('Distribuição de Acurácia por Método', fontweight='bold')
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Tabela de resultados
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = []
        for method, acc_mean, acc_std in zip(methods, accuracy_means, accuracy_stds):
            table_data.append([method, f'{acc_mean:.4f}', f'{acc_std:.4f}'])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Método', 'Acurácia Média', 'Desvio Padrão'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.5, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Estilo da tabela
        for i in range(len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGráficos salvos em: {save_path}")
        
        return fig
    
    def generate_report(self):
        """
        Gera um relatório textual com os resultados da análise.
        """
        if not self.results:
            print("Execute run_all_methods() primeiro!")
            return
        
        print("\n" + "="*80)
        print("RELATÓRIO COMPARATIVO DE MÉTODOS DE VALIDAÇÃO CRUZADA")
        print("Classificador SVM para Predição de Doenças")
        print("="*80)
        
        print(f"\nDataset: {self.X.shape[0]} amostras, {self.X.shape[1]} características")
        print(f"Distribuição de classes: {np.bincount(self.y)}")
        
        print("\n" + "-"*80)
        print("RESULTADOS POR MÉTODO")
        print("-"*80)
        
        for key, result in self.results.items():
            print(f"\nMétodo: {result['method']} (n_splits={result['n_splits']})")
            print(f"  Accuracy:  {result['accuracy_mean']:.4f} (+/- {result['accuracy_std']:.4f})")
            if result['precision_mean'] is not None:
                print(f"  Precision: {result['precision_mean']:.4f} (+/- {result['precision_std']:.4f})")
                print(f"  Recall:    {result['recall_mean']:.4f} (+/- {result['recall_std']:.4f})")
                print(f"  F1-Score:  {result['f1_mean']:.4f} (+/- {result['f1_std']:.4f})")
        
        print("\n" + "-"*80)
        print("ANÁLISE COMPARATIVA")
        print("-"*80)
        
        # Encontra melhor método por acurácia
        best_method = max(self.results.items(), key=lambda x: x[1]['accuracy_mean'])
        print(f"\nMelhor método (Acurácia): {best_method[1]['method']}")
        print(f"  Acurácia: {best_method[1]['accuracy_mean']:.4f} (+/- {best_method[1]['accuracy_std']:.4f})")
        
        # Compara variabilidade
        print("\nVariabilidade (Desvio Padrão da Acurácia):")
        sorted_by_std = sorted(self.results.items(), key=lambda x: x[1]['accuracy_std'])
        for key, result in sorted_by_std:
            print(f"  {result['method']:25s}: {result['accuracy_std']:.4f}")
        
        print("\n" + "="*80)


def main():
    """
    Função principal para executar a análise.
    """
    print("="*80)
    print("ANÁLISE COMPARATIVA DE MÉTODOS DE VALIDAÇÃO CRUZADA")
    print("Classificador SVM para Predição de Doenças Cardíacas")
    print("="*80)
    
    # Inicializa análise
    analysis = SVMCrossValidationAnalysis(random_state=42)
    
    # Carrega dados
    analysis.load_data()
    
    # Executa todos os métodos
    # Nota: include_loo=True pode ser muito lento para datasets grandes
    analysis.run_all_methods(k_values=[5, 10], include_loo=False)
    
    # Gera visualizações
    analysis.create_comparison_plots('cv_comparison.png')
    
    # Gera relatório
    analysis.generate_report()
    
    print("\nAnálise concluída com sucesso!")


if __name__ == "__main__":
    main()
