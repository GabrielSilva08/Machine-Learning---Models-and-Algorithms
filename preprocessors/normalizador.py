import numpy as np

class Normalizador:
    def __init__(self, X, method="MinMaxScaler"):
        """Método construtor da classe Normalizador

        Args:
            X (_type_): Dados a serem normalizados/desnormalizados
            method (str, optional): Técnica de normalização especificada. Por padrão, MinMaxScaler (X -> [0, 1])
        """
        if method == "MinMaxScaler":
            self.method = method
            self.Xmin = X.min(axis=0)
            self.Xmax = X.max(axis=0)

    def normaliza(self, X):
        """Método que normaliza os dados de X

        Args:
            X (_type_): Dados a serem normalizados

        Returns:
            X_normalizado (_type_): Dados de X já normalizados
        """
        if self.method == "MinMaxScaler":
            X = (X - self.Xmin)/(self.Xmax - self.Xmin)
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
            return X