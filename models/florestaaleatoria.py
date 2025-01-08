import numpy as np
from models.arvorededecisao import ArvoreDeDecisao

class FlorestaAleatoria:
    def __init__(self, n_modelos_base=1, indice_de_impureza="gini", impureza_minima=0, profundidade_maxima=None, random_state=-1):
        """Método construtor que inicializa o modelo de floresta aleatória.

        Args:
            n_modelos_base (int, optional): Número de árvores de decisão. Defaults to 1.
            indice_de_impureza (str, optional): Nome do índice de impureza aplicado a todas as árvores. Defaults to "gini".
            impureza_minima (int, optional): Índice de impureza aplicado a todas as árvores. Defaults to 0.
            profundidade_maxima (_type_, optional): Profundidade a qual todas as árvores devem se limitar. Defaults to None.
            random_state (int, optional): Parâmetro necessário para garantir reproducibilidade do modelo, utilizado na etapa de bagging. Defaults to -1 (não reprodutível).
        """
        self.n_modelos_base = n_modelos_base
        self.indice_de_impureza= indice_de_impureza
        self.impureza_minima = impureza_minima
        self.profundidade_maxima = profundidade_maxima
        self.random_state = random_state
        self.arvores = None

    def ajuste(self, X_train, y_train):
        """Método que realiza o treinamento do modelo de floresta aleatória.

        Args:
            X_train (_type_): Dados de treinamento de entrada
            y_train (_type_): Dados de treinamento de saída
        """
        self.arvores = []
        L = self.bagging(X_train, y_train)
        for i in range(self.n_modelos_base):
            tree = ArvoreDeDecisao(indice_de_impureza=self.indice_de_impureza, impureza_minima=self.impureza_minima, profundidade_maxima=self.profundidade_maxima)
            Xy_train = L[i]
            tree.ajuste(Xy_train[:, :-1], Xy_train[:, [-1]])
            self.arvores.append(tree)

    def prever(self, X_test):
        """Método que realiza a predição de novos valores. 
           É realizado um comitê por votação majoritária.

        Args:
            X_test (_type_): Dados de teste de entrada

        Returns:
            y_pred (_type_): Vetor com os valores de saída preditos.
        """
        Y_pred_arvores = np.empty((X_test.shape[0], self.n_modelos_base), dtype=np.int8)
        y_pred_arvores = np.empty((X_test.shape[0], 1))
        for i in range(self.n_modelos_base):
            aux = self.arvores[i].prever(X_test)
            for j in range(X_test.shape[0]):
                Y_pred_arvores[j][i] = aux[j]
        for i in range(X_test.shape[0]):
            y_pred_arvores[i] = np.bincount(Y_pred_arvores[i]).argmax()
        return y_pred_arvores
    
    def prever_proba(self, X_test):
        """Método que realiza o predict_proba para cada árvore presente, e retorna a
           média de suas probabilidades.

        Args:
            X_test (_type_): Dados de teste de entrada

        Returns:
            y_proba (_type_): Vetor de probabilidades de saída preditos.
        """
        Y_proba_arvores = np.empty((X_test.shape[0], self.n_modelos_base))
        for i in range(self.n_modelos_base):
            aux = self.arvores[i].prever_proba(X_test)
            for j in range(X_test.shape[0]):
                Y_proba_arvores[j][i] = aux[j]
        return Y_proba_arvores.mean(axis=1).reshape(-1,1)

    def bagging(self, X_train, y_train):
        """Método auxiliar que realiza o bootstrap aggregating. Possibilitando a criação de modelos
           que apresentem erros distintos, com isso, garantindo diversidade.

        Args:
            X_train (_type_): Dados de treinamento de entrada
            y_train (_type_): Dados de treinamento de saída

        Returns:
            L (list): Lista de n_modelos_base amostras com reposição.
        """
        N = X_train.shape[0]
        p = int(60/100 * N)
        idx = np.arange(0, N)
        if self.random_state != -1:
            np.random.seed(self.random_state)
        L = []
        Xy_train = np.hstack([X_train, y_train])
        for i in range(self.n_modelos_base):
            l = np.random.choice(idx, size=p, replace=True)
            L.append(Xy_train[l])
        return L