import numpy as np
from utils.utils import dist as d

class Kmedias:
    def __init__(self, k, ini="aleatoria", funcao_de_dissimilaridade="euclidiana", maxima_inter=1000, random_state=-1):
        """Método construtor do modelo K-Médias.

        Args:
            k (int): Número de clusters (grupos)
            ini (str): Forma de inicialização dos centróides
                      ['aleatoria', 'k-means++']
            funcao_de_dissimilaridade (str): Métrica adotada para computar as partições
                                            ['euclidiana', 'manhattan', 'mahalanobis']
            maxima_inter (int): Máximo número de interações do recálculo dos centróides
            random_state (int, optional): Semente aleatória utilizada na inicialização dos centróides. Por padrão, -1 (não reproduzível).
        """
        self.k = k
        self.ini = ini
        self.funcao_de_dissimilaridade = funcao_de_dissimilaridade
        self.maxima_inter = maxima_inter
        self.random_state=random_state
        # Matriz cujo os vetores linhas compõem os k centróides
        self.centroides = None 
        # Lista de matrizes onde cada uma compõem os padrões pertecentes ao cluster i
        self.clusters = None
        # vetor de clusters, a qual para o padrão i, seu cluster é c[i]
        self.c = None
        # Função custo dos clusters
        self.erro_de_reconstrucao = 0

    def ajuste(self, X):
        """Método que realiza o treinamento do modelo.

        Args:
            X (_type_): Dados os quais se deseja realizar o clustering
        """
        # Número de padrões
        self.N = X.shape[0]
        # Número de atributos
        self.D = X.shape[1]
        # Média dos dados
        self.U = np.mean(X, axis=0)
        # Matriz de covariância dos dados
        self.S = X - self.U
        aux = np.empty((self.D, self.D))
        for i in range(self.k):
            aux += self.S[i].T@self.S[i]
        self.S = aux * 1/(self.k - 1)
        # Vetor indicativo dos clusters
        self.c = np.empty(self.N, dtype=np.int8)

        # Algoritmo de Lloyd
        # Cálculo dos centróides
        if self.ini == "aleatoria":
            if self.random_state != -1:
                np.random.seed(self.random_state)
            idxs = np.random.choice(self.N, size=self.k, replace=False)
            self.centroides = X[idxs].astype(np.float64)
        elif self.ini == "k-means++":
            if self.random_state != -1:
                np.random.seed(self.random_state)
            idx = np.random.choice(self.N)
            self.centroides = X[idx].astype(np.float64)
            dist = np.empty(self.N)
            for i in range(self.k-1):
                for j, x in enumerate(X):
                    dist[j] = self.distancia2(x).min()
                novo_centroide = X[dist.argmax()]
                self.centroides = np.vstack([self.centroides, novo_centroide])

        for n in range(self.maxima_inter):
            # Inicialização do espaço de memória dos clusters
            self.clusters = []
            for i in range(self.k):
                self.clusters.append(np.zeros((1, self.D)))
            # Cálculo das partições
            for i in range(self.N):
                self.c[i] = self.distancia2(X[i]).argmin()
                # Alocar o padrão ao cluster correspondente
                self.clusters[self.c[i]] = np.vstack([self.clusters[self.c[i]], X[i]])
            # remoção do vetor inicial utilizado para reservar o espaço de memória
            for i in range(self.k):
                self.clusters[i] = np.delete(self.clusters[i], 0, axis=0)
            # Recalculo dos centróides de cada cluster
            for i in range(self.k):
                self.centroides[i] = self.clusters[i].mean(axis=0)
        # Cálculo do erro de recontrução
        for i, C in enumerate(self.clusters):
            for x in C:
                self.erro_de_reconstrucao += d(x, self.centroides[i]) ** 2

    def prever(self, X):
        """Método que realiza o agrupamento de novos dados.

        Args:
            X (_type_): Padrões novos

        Returns:
            c_pred (_type_): Clusters a qual cada padrão mais se assemelha
        """
        c_pred = np.empty(X.shape[0])
        for i, x in enumerate(X):
            c_pred[i] = self.distancia2(x).argmin()
        return c_pred

    def distancia2(self, x):
        """Método auxiliar responsável pelo cálculo da distância (definida pelo atributo 'funcao_de_dissimilaridade') do padrão x 
        com os centróides.

        Args:
            x (_type_): Padrão de entrada

        Raises:
            Exception: Exceção lançada quando a funcao_de_dissimilaridade não foi especificada ou definida.

        Returns:
            distancias2 (_type_): Vetor de distâncias quadradas (caso 'funcao_de_dissimilaridade' seja 'euclidiana' ou 'mahalanobis')
            do padrão x com todos os centróides.
        """
        if self.funcao_de_dissimilaridade == "euclidiana":
            if self.centroides.ndim == 1:
                return ((x - self.centroides) ** 2).sum()
            return ((x - self.centroides) ** 2).sum(axis = 1).reshape(-1, 1)            
        if self.funcao_de_dissimilaridade == "manhattan":
            if self.centroides.ndim == 1:
                return (np.abs(x - self.centroides)).sum()
            return (np.abs(x-self.centroides)).sum(axis=1).reshape(-1, 1)
        if self.funcao_de_dissimilaridade == "mahalanobis":
            d2m = np.empty(1 if self.centroides.ndim == 1 else len(self.centroides))
            # Computando a distância quadrática pra cada centróide
            for i in range(len(d2m)):
                if len(d2m) == 1:
                    d2m[i] = (x-self.centroides)@(np.linalg.inv(self.S + 0.000001*np.eye(self.D)))@(x-self.centroides).T
                else:
                    d2m[i] = (x-self.centroides[i])@(np.linalg.inv(self.S + 0.000001*np.eye(self.D)))@(x-self.centroides[i]).T
            return d2m.reshape(-1,1)
        raise Exception("'metric' não foi definida")