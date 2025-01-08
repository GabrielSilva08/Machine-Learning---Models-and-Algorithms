import numpy as np
from utils.utils import *

# Análise de Discriminante Gaussiano
# Priori -> Relativo ao número de instâncias da classe
# Verosimilhança -> Distribuição Gaussiana
class Analise_de_Discriminante_Gaussiano:
    def __init__(self, K, priori="relativa"):
        """ Método construtor do classificador

        Args:
            K (int): Número de classes
            priori (str, optional): Modo como serão computadas as prioris ("equiprovavel" ou "relativa"). Por padrão, relativa.
                equiprovável: Todas as priori assumem um valor padrão de 1/K.
                relativa: As prioris de uma classe são proporcionais à quantidade de padrões dessa mesma classe.
        """
        # Número de classes
        self.K = K
        # Modo como será computado o vetor de prioris
        self.priori=priori
        # Vetor de prioris p (1 x K)
        self.p = None
        # Matriz de médias U (K x D)
        self.U = None
        # Matriz de matrizes de covariância (D x D x K)
        self.S = None
        # Escalar correspondente ao número de atributos
        self.D = None
        
    def ajuste(self, X_train, y_train):
        """Método que realiza o treinamento do módelo.

        Args:
            X_train (_type_): Dados de treinamento de entrada
            y_train (_type_): Dados de treinamento de saída

        Raises:
            ZeroDivisionError: Erro retornado quando o número de padrões disponibilizados para treinamento é de apenas 1 (1/(N - 1) = 1/0)
        """
        # Computação do vetor de prioris p
        self.D = X_train.shape[1]
        if self.priori == "relativa":
            # p[i] = Ni/N, onde i pertence a [0, K)
            if self.K == 2:
                self.p = np.unique(y_train, return_counts=True)[1] * 1/np.sum(np.unique(y_train, return_counts=True)[1])
            else:
                self.p = np.unique(np.argmax(y_train, axis=1), return_counts=True)[1] * 1/np.sum(np.unique(np.argmax(y_train, axis=1), return_counts=True)[1])
        elif self.priori == "equiprovavel":
            # p[i] = 1/N, para todo i pertecente a [0, k)
            self.p = np.full((self.K), 1/self.K)

        # Computando as médias dos padrões por classe e guardando numa matriz de médias U (K x D)
        self.U = np.empty((self.K, self.D))
        # Vetor gerado por classe
        for i in range(self.K):
            # Padrões pertecentes à classe i
            X = X_train[self.filter_by_class(i, y_train)]
            # U[i,:] -> 1 x D
            self.U[i, :] = np.mean(X, axis=0)

        # Computando as matrizes de covariância dos padrões por classe e guardando numa matriz de matrizes de covariância S (D x D x K)
        self.S = np.empty((self.D, self.D, self.K))
        # Matriz gerada por classe
        for i in range(self.K):
            # Padrões pertecentes à classe i
            X = X_train[self.filter_by_class(i, y_train)]
            if X.shape[0] == 1:
                raise ZeroDivisionError("Alguma das classes de treino possui o número de instâncias de apenas 1.")
            # S[:,:,i] -> D x D
            self.S[:,:,i] = 1/(X.shape[0] - 1) * self.sum_by_class(X, self.U[i, :])


    def prever(self, X_test):
        """Método que retorna as predições para novos padrões.

        Args:
            X_test (_type_): Dados de entrada a qual se quer realizar novas predições

        Returns:
            y_pred (_type_): Predição sobre os dados de saída
        """
        n = X_test.shape[0]
        # p(Ci | x), onde i pertence a [0, K) 
        Y_pred = np.empty((n, self.K))
        # Computando log de posterioris por classe e por padrão de entrada
        for i in range(self.K):
            termos_constantes = -1/2 * np.log(np.linalg.det(self.S[:,:,i] + 0.0001*np.eye(self.D))) + np.log(self.p[i])
            for j in range(n):
                # Subtração da norma de Mahalanobis
                Y_pred[j, i] = termos_constantes - (0.5 * ((X_test[j, :] - self.U[i, :]).T) @ np.linalg.inv(self.S[:,:,i] + 0.0001*np.eye((self.D))) @ (X_test[j, :] - self.U[i,:]))
        # Selecionando as classes de maior posteriori
        y_pred = np.argmax(Y_pred, axis=1)
        return y_pred
    
    def filter_by_class(self, k,  y_train):
        """Método auxiliar que realiza a filtragem de um conjunto de dados pertecentes à classe k.

        Args:
            k (int): Classe em que se desejar filtrar os dados.
            y_train (_type_): Dados de saída correspondentes à matriz em que se deseja realizar a filtração.

        Returns:
            _type_: y_train filtrado por k.
        """
        if y_train.shape[1] == 1:
            return [True if y == k else False for y in y_train]
        return [True if y == k else False for y in ((np.argmax(y_train, axis=1)).reshape(-1,1))]
    
    def sum_by_class(self, X, u):
        """Método auxiliar que realiza o somatório de padrões já filtrados por classe.

        Args:
            X (_type_): Padrões de entradas filtrados à uma classe
            u (_type_): Vetor de médias de uma dada classe

        Returns:
            _type_: Matriz resultante do somatório dos padrões mediante realização do produto interno
        """
        u = u.reshape(1, -1)
        n = X.shape[0]
        aux = ((X[0]-u).T)@(X[0]-u)
        for i in range(1, n):
            aux += ((X[i]-u).T)@(X[i]-u)
        return aux

def grid_search_ADG(search_space, Xy_train, k, c):
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

    for p in search_space["priori"]:
        # Avaliação de uma combinação de hiperparâmetros
        classificador_adg = Analise_de_Discriminante_Gaussiano(c, p)
        folds = kfold(Xy_train, k)
        acuracias = []
        # k fold
        for f in range(k):
            valid_fold = folds.pop()
            train_fold = np.vstack(folds)
            if c == 2:
                classificador_adg.ajuste(train_fold[:,:-1], train_fold[:,[-1]])
                y_pred = classificador_adg.prever(valid_fold[:,:-1])
                acuracias.append(acc(valid_fold[:,[-1]], y_pred)[0])
            else:
                classificador_adg.ajuste(train_fold[:, :-c], train_fold[:, -c:])
                y_pred = classificador_adg.prever(valid_fold[:,:-c])
                acuracias.append(acc(np.argmax(valid_fold[:, -c:], axis=1).reshape(-1,1), y_pred.reshape(-1,1))[0])
            folds.insert(0, valid_fold)
        acuracia_media = np.array(acuracias).mean()
        # Salvando o resultado da melhor combinação
        if acuracia_media > melhor_acuracia_media:
            melhor_acuracia_media = acuracia_media
            melhores_hiperparametros = (p,)
    return melhor_acuracia_media, melhores_hiperparametros

