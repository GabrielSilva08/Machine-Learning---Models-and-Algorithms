import numpy as np
import warnings
warnings.filterwarnings("ignore")
from utils.utils import *

class RegressaoLogistica:
    def __init__(self, alpha=0.01, l=0, threshold=0.5):
        """Método construtor do classificador.

        Args:
            alpha (float): Passo de aprendizado
            l (float): Hiperparâmetro de regularização L2 lambda
            threshold (float) : Limiar de decisão da classificação
        """
        self.alpha = alpha
        self.l = l
        self.w = None
        self.e = None
        self.threshold = threshold
        
    def ajuste(self, X_train, y_train):
        """Método que realiza o treinamento do modelo.

        Args:
            X_train (_type_): Dados de treinamento de entrada
            y_train (_type_): Dados de treinamento de saída

        Returns:
            w (_type): Vetor de parâmetros gerado durante o treinamento
            e (_type): Vetor de erros gerado na última interação do treinamento
        """
        self.w, self.e = algoritmo_GD(X_train, y_train, self.alpha, self.l, "logistic")

    def prever(self, X_test):
        """Método que retorna as predições para novos padrões.

        Args:
            X_test (_type_): Dados de entrada a qual se quer realizar novas predições

        Returns:
            y_pred (_type_): Predição sobre os dados de saída
        """
        y_proba = self.prever_proba(X_test)
        y_pred = (y_proba >= self.threshold).astype("int8")
        return y_pred
    
    def prever_proba(self, X_test):
        """Método que retorna a probabilidade das predições para novos padrões.

        Args:
            X_test (_type_): Dados de entrada a qual se quer realizar novas predições

        Returns:
            y_proba (_type): Probabilidade das predições sobre os dados de saída 
        """
        X_test = np.hstack([np.ones((X_test.shape[0],1)), X_test])
        y_proba = sig(X_test@self.w)
        return y_proba


def sig(z):
    """Função que retorna o sigmóide de z

    Args:
        z (_type_): Valor a qual se deseja computar a sigmóide

    Returns:
        _type_: sigmóide de z (1/(1 + e^{-z}))
    """
    return 1/(1+np.exp(-z))

def algoritmo_GD(X, y, alpha=0.01, l=0, model="linear"):
    """Algoritmo Gradiente Descendente.

        Args:
            X (_type_): Dados de entradas
            y (_type_): Dados de saídas
            alpha (float): Passo de aprendizado
            l (float): Hiperparâmetro de regularização L2 lambda
            model (string): Especificação da função custo a ser otimizada.
                            ["linear" (default), "logistic"]
            
    Rreturns:
            w (_type_): Vetor de parâmetros
            e (_type_): Vetor de erros (sendo esse obtido após o término da última época)
    """
    # Coluna de 1s à esquerda para ter a computação do parâmetro w0
    X = np.hstack([np.ones((X.shape[0],1)), X])
    # Inicialização dos D+1 parâmetros
    w = np.zeros((X.shape[1])).reshape(-1,1)
    t = 0
    while t < 10000:
        # Atualização da interação
        t += 1
        # Cálculo do erro
        y_chapeu = X@w
        if model == "logistic":
            e = y - sig(y_chapeu)
        else:
            e = y - y_chapeu
        # Atualização dos parâmetros
        for i in range(w.size):
            w[i] = w[i] + alpha*(np.mean(e * X[:,[i]]) - l*w[i])
    return w, e

def grid_search_RL(search_space, Xy_train, k):
    """Função que realiza o grid search via k-fold cross validation.

    Args:
        search_space (dict): Dicionário contendo pra uma lista de candidatos para cada hiperparâmetro.
        Xy_train (_type_): Dados de treinamento
        k (_type_): Número de folds

    Returns:
        melhor_acuracia_media (float): Melhor acurácia média obtida por uma combinação de hiperparâmetros.
        melhores_hiperparâmetros (tuple): Hiperparâmetros que permitiram computação da melhor média.
    """
    melhor_acuracia_media = 0 
    melhores_hiperparametros = None

    for i in search_space["alpha"]:
        for j in search_space["lambda"]:
            # Avaliação de uma combinação de hiperparâmetros
            classificador_rl = RegressaoLogistica(i,j)
            folds = kfold(Xy_train, k)
            acuracias = []
            # k fold
            for f in range(k):
                valid_fold = folds.pop()
                train_fold = np.vstack(folds)
                classificador_rl.ajuste(train_fold[:,:-1], train_fold[:,[-1]])
                y_pred = classificador_rl.prever(valid_fold[:,:-1])
                acuracias.append(acc(valid_fold[:,[-1]], y_pred)[0])
                folds.insert(0, valid_fold)
            acuracia_media = np.array(acuracias).mean()
            # Salvando o resultado da melhor combinação
            if acuracia_media > melhor_acuracia_media:
                melhor_acuracia_media = acuracia_media
                melhores_hiperparametros = (i, j)
    return melhor_acuracia_media, melhores_hiperparametros