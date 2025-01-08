import numpy as np

class PCA:
    def __init__(self, n_de_componentes=2, l=10**-5):
        """Método construtor do modelo PCA.

        Args:
            n_de_componentes (int, optional): Número de componentes principais. Por padrão, 2.
        """
        self.n_de_componentes = n_de_componentes
        self.l = l
        self.variancias_projetadas = None
        # Matriz de projeção
        self.P = None
        # Soma das variâncias projetadas
        self.variancia_explicada = None
        
    def ajuste(self, X):
        """Método que realiza o cálculo da matriz de projeção (autovetores de Sigma),
           bem como o cálculo das variâncias projetadas (autovalores de Sigma).

        Args:
            X (_type_): Padrões de entrada os quais se deseja realizar a redução de sua dimensionalidade.

        Raises:
            Exception: Lançada quando usuário especifica um número de projeções maior que o número de atributos de X.
        """
        N, D = X.shape
        # Verificação da quantidade de componentes
        if self.n_de_componentes > D:
            raise Exception("Número de componentes desejadas superior ao número de atributos!")
        # Cálculo da matriz de covariância dos dados de X
        u = X.mean(axis=0).reshape(1,-1)
        aux = np.zeros((D, D))
        for x in X:
            aux += ((x-u).T)@(x-u)
        Sigma = aux/(N-1)
        # Cálculo dos autovalores e autovetores por meio da decomposição em valores singulares
        U, S, Vt = np.linalg.svd(Sigma-self.l*np.eye(D))
        self.P = U.T[:self.n_de_componentes]
        self.variancias_projetadas = S[:self.n_de_componentes]
        self.variancia_explicada = self.variancias_projetadas.sum()

    def prever(self, X):
        """Método que realiza a projeção dos padrões de X

        Args:
            X (_type_): Padrões de entrada os quais se deseja realizar a redução de sua dimensionalidade.

        Returns:
            Z (_type_): Padrões de X com as dimensões reduzidas
        """
        if self.n_de_componentes == 1:
            return X@(self.P.reshape(-1,1))
        return X@self.P.T
