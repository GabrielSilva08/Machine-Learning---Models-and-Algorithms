import numpy as np

class Normalizador:
    def __init__(self, X, method="MinMaxScaler"):
        """Método construtor da classe Normalizador

        Args:
            X (_type_): Dados a serem normalizados/desnormalizados
            method (str, optional): Técnica de normalização especificada. Por padrão, MinMaxScaler (X -> [0, 1]).
                                    Valores aceitos -> ["MinMaxScaler" (X -> [0,1]), "StandardScaler" (média = 0 e desvio_padrão = 1)]
        """
        if method == "MinMaxScaler":
            self.method = method
            self.Xmin = X.min(axis=0)
            self.Xmax = X.max(axis=0)
        elif method == "StandardScaler":
            self.method = method
            self.u = X.mean(axis=0)
            self.d = X.std(axis=0, ddof=1)

    def normaliza(self, X):
        """Método que normaliza os dados de X

        Args:
            X (_type_): Dados a serem normalizados

        Returns:
            X_normalizado (_type_): Dados de X já normalizados
        """
        if self.method == "MinMaxScaler":
            X = (X - self.Xmin)/(self.Xmax - self.Xmin)
        if self.method == "StandardScaler":
            if np.any(np.isclose(0, self.d)) == True:
                raise ZeroDivisionError("Desvio padrão próximo de zero, impossibilidade de normalização via z-score.")
            X = (X - self.u)/self.d
        return X
    
    def desnormaliza(self, X):
        """Método que desnormaliza os dados de X

        Args:
            X (_type_): Dados a serem desnormalizados

        Returns:
            X_desnormalizado (_type_): Dados de X já desnormalizados
        """
        if self.method == "MinMaxScaler":
            X = X * (self.Xmax - self.Xmin) + self.Xmin
        if self.method == "StandardScaler":
            X = X * self.d + self.u
        return X