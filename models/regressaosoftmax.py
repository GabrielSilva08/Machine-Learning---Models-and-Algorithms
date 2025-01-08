import numpy as np
from utils.utils import *

class RegressaoSoftmax:
    def __init__(self, alpha=0.01, l=0):
        """Método construtor do classificador.

        Args:
            alpha (float): Passo de aprendizado
            l (float): Hiperparâmetro de regularização L2 lambda
        """
        self.alpha = alpha
        self.l = l
        # Número de classes
        self.K = None
        # Matriz de parâmetros
        self.W = None
        # Matriz de erros
        self.ERRO = None

    def ajuste(self, X_train, y_train):
        """Método que realiza o treinamento do modelo.

        Args:
            X_train (_type_): Dados de treinamento de entrada
            y_train (_type_): Dados de treinamento de saída
        """
        self.K = y_train.shape[1]
        self.W, self.ERRO = algoritmo_GD_multivariado(X_train, y_train, self.alpha, self.l)

    def prever(self, X_test):
        """Método que retorna as predições para novos padrões.

        Args:
            X_test (_type_): Dados de entrada a qual se deseja realizar as classificações
        Returns:
            y_pred (_type_): Classificação sobre os dados de saída
        """
        Y_proba = self.prever_proba(X_test)
        y_pred = np.argmax(Y_proba, axis=1)
        return y_pred.reshape(-1, 1)
    
    def prever_proba(self, X_test):
        """Método que retorna a probabilidade das predições associadas a cada classe.

        Args:
            X_test (_type_): Dados de entrada a qual se deseja realizar novas predições

        Returns:
            Y_proba (_type_): Matriz de probabilidade das predições associadas a cada classe
        """
        n = X_test.shape[0]
        X_test = np.hstack([np.ones((X_test.shape[0],1)), X_test])
        Y_proba = np.empty((n, self.K))
        for i in range(n):
            for j in range(self.K):
                Y_proba[i, j] = softmax(X_test[i], self.W, j)[0]
        return Y_proba
    
def softmax(x, W, k):
    """Função responsável por computar a softmax.

    Args:
        x (_type_): Instância dos dados de entrada (1, N)
        W (_type_): Matriz de parâmetros (D, K)
        k (_type_): Classe especificada

    Returns:
        y_chapeu_k (_type_): Escalar associado a intância x classe k
    """
    return np.exp(W[:, [k]].T@x)/np.sum(np.exp(x@W))

def algoritmo_GD_multivariado(X, Y, alpha=0.01, l=0):
    """Algoritmo Gradiente Descendente adaptado para classificação multiclasse.

        Args:
            X (_type_): Dados de entradas
            y (_type_): Dados de saídas
            alpha (float): Passo de aprendizado
            l (float): Hiperparâmetro de regularização lambda
            
        returns:
            W (_type_): Matriz de parâmetros (D x K)
            E (_type_): Matriz de erros (N x K)
    """
    # Coluna de 1s à esquerda para ter a computação do parâmetro w0
    X = np.hstack([np.ones((X.shape[0],1)), X])
    N = X.shape[0]
    K = Y.shape[1]
    E = np.empty((N, K))

    # Inicialização da matrix de parâmetros (D+1, k) 
    W = np.zeros((X.shape[1], K))
    t = 0
    while t < 10000:
        # Atualização da interação
        t += 1

        # Cálculo do erro
        for i in range(N):
            for k in range(K):
                E[i, k] = Y[i, k] - softmax(X[i], W, k)[0]

        # Atualização dos parâmetros
        for k in range(K):
            for i in range(W.shape[0]):
                W[i, k] = W[i, k] + alpha*np.mean((E[:, [k]]  *  X[:, [i]]) - l*W[i, k])
    return W, E

def grid_search_RS(search_space, Xy_train, k, c):
    """Função que realiza o grid search via k-fold cross validation.

    Args:
        search_space (dict): Dicionário contendo pra uma lista de candidatos para cada hiperparâmetro.
        Xy_train (_type_): Dados de treinamento
        k (int): Número de folds
        c (int): Número de classes

    Returns:
        melhor_acuracia_media (float): Melhor acurácia média obtida por uma combinação de hiperparâmetros.
        melhores_hiperparâmetros (tuple): Hiperparâmetros que permitiram computação da melhor média.
    """
    melhor_acuracia_media = 0 
    melhores_hiperparametros = None

    for i in search_space["alpha"]:
        for j in search_space["lambda"]:
            # Avaliação de uma combinação de hiperparâmetros
            classificador_rs = RegressaoSoftmax(i,j)
            folds = kfold(Xy_train, k)
            acuracias = []
            # k fold
            for f in range(k):
                valid_fold = folds.pop()
                train_fold = np.vstack(folds)
                classificador_rs.ajuste(train_fold[:,:-c], train_fold[:, -c:])
                y_pred = classificador_rs.prever(valid_fold[:,:-c])
                acuracias.append(acc(np.argmax(valid_fold[:, -c:], axis=1), y_pred)[0])
                folds.insert(0, valid_fold)
            acuracia_media = np.array(acuracias).mean()
            # Salvando o resultado da melhor combinação
            if acuracia_media > melhor_acuracia_media:
                melhor_acuracia_media = acuracia_media
                melhores_hiperparametros = (i, j)
    return melhor_acuracia_media, melhores_hiperparametros
