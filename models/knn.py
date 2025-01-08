import numpy as np

class kNN:
    def __init__(self, k=3, metric="euclidiana"):
        """Método construtor do modelo

        Args:
            k (int, optional): Número de vizinhos a considerar na classificação. Por padrão, 3.
            metric (str, optional): Métrica de similaridade adotada. Por padrão, "euclidiana".
                                    Valores aceitos -> ["euclidiana", "manhattan", "mahalanobis"]
        """
        self.k = k
        self.metric = metric
        # Número de padrões dos dados de treinamento
        self.N = None
        # Número de atributos
        self.D = None
        # Conjunto de treinamento
        self.X_train = None
        self.y_train = None
        # Atributos computados caso a métrica especificada seja a de "Mahalanobis"
        # Vetor de média dos atributos dos dados de treinamento
        self.U = None
        # Matriz de covariância
        self.S = None

    def ajuste(self, X_train, y_train):
        """Método que realiza o treinamento do modelo.

        Args:
            X_train (_type_): Dados de treinamento de entrada
            y_train (_type_): Dados de treinamento de saída
        """
        self.N = X_train.shape[0]
        self.D = X_train.shape[1]
        self.X_train = X_train
        self.y_train = y_train

    def prever(self, X_test):
        """Método que realiza a predição para novos padrões.

        Args:
            X_test (_type_): Padrões que se desejam realizar novas predições

        Returns:
            y_pred (_type_): Vetor com os valores de saída preditos.
        """
        N_test = X_test.shape[0]
        y_pred = np.empty(N_test)
        # Predição realizada para cada padrão presente em X_test
        for i in range(N_test):
            distancias2 = self.distancia2(X_test[i])
            distancias2 = np.hstack([distancias2, self.y_train])
            distancias2 = distancias2[distancias2[:, 0].argsort()]
            count = distancias2[:self.k, [1]]
            classe, idx =  np.unique(count, return_counts=True)
            y_pred[i] = classe[np.argmax(idx)]
        return y_pred
    
    def distancia2(self, x):
        """Método auxiliar responsável pelo cálculo da distância (definida pelo atributo 'metric') do padrão x 
        com os padrões presentes no conjunto de treinamento.

        Args:
            x (_type_): Novo padrão

        Raises:
            Exception: Exceção lançada quando a métrica não foi especificada ou definida.

        Returns:
            distancias2 (_type_): Vetor de distâncias quadradas (caso 'metric' seja 'euclidiana' ou 'mahalanobis')
            do padrão x com todos os padrões presentes no conjunto de treinamento.
        """
        if self.metric == "euclidiana":
            return ((x - self.X_train) ** 2).sum(axis = 1).reshape(-1, 1)            
        if self.metric == "manhattan":
            return (np.abs(x-self.X_train)).sum(axis=1).reshape(-1, 1)
        if self.metric == "mahalanobis":
            self.U = np.mean(self.X_train, axis=0)
            self.S = self.X_train - self.U
            aux = np.empty((self.D, self.D))
            for i in range(self.N):
                aux += self.S[i].T@self.S[i]
            self.S = aux * 1/(self.N - 1)
            d2m = np.empty(self.N)
            # Computando a distância quadrática pra cada padrão presente no treinamento
            for i in range(self.N):
                d2m[i] = (x-self.X_train[i])@(np.linalg.inv(self.S + 0.0001*np.eye(self.D)))@(x-self.X_train[i]).T
            return d2m.reshape(-1,1)
        raise Exception("'metric' não foi definida")
