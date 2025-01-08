import numpy as np

class MatrizDeConfusaoBinaria:
    def __init__(self, y_real, y_pred):
        """Método construtor que realiza o cálculo dos valores presentes na matriz de confusão binária. Sendo:
        
        0 -> Classe Negativa
        1 -> Classe Positiva

        Args:
            y_real (_type_): Dados de saída reais
            y_pred (_type_): Dados de saída preditos por um classificador
        """
        # Verdadeiro Negativo (True Negative)
        self.TN = np.count_nonzero((y_real + y_pred) == 0)
        # Falso Positivo (False Positive)
        self.FP = np.count_nonzero((y_real - y_pred) == -1)
        # Falso Negativo (False Negative)
        self.FN = np.count_nonzero((y_real - y_pred) == 1)
        # Verdadeiro Positivo (True Positive)
        self.TP = np.count_nonzero((y_real + y_pred) == 2)
        # Matriz de confusão
        self.matriz = np.array([[self.TN, self.FP],[self.FN, self.TP]])
    def __str__(self):
        """Método responsável por retornar a matriz de confusão.

        Returns:
            str: Matriz de confusão
        """
        return str(self.matriz)
    def acuracia(self):
        """Método que cálcula a acurácia a partir dos dados de saídas fornecidos.

        Returns:
            float: Acurácia -> (TP+TN)/(TN+FP+FN+TP)
        """
        return (self.TP+self.TN)/(self.TP+self.TN+self.FN+self.FP)
    def precisao(self):
        """Método que cálcula a precisão a partir dos dados de saídas fornecidos.

        Returns:
            float: Precisão -> TP/(TP+FP)
        """
        return self.TP/(self.TP+self.FP)
    def revocacao(self):
        """Método que cálcula a revocação (recall) a partir dos dados de saídas fornecidos.

        Returns:
            float: Revocação -> TP/(TP+FN)
        """
        return self.TP/(self.TP+self.FN)
    def fb_score(self, b=1):
        """Método que cálcula o F beta-score a partir dos dados de saídas fornecidos.

        Args:
            b (int, optional): Valor de beta, sendo esse informando o quão importante é a revocação comparada com a precisão. Por padrão, 1 (F1-score).

        Returns:
            float: Fb-score -> (1 + b²) * revocação * precisão/(revocação + b² * precisão) 
        """
        r = self.revocacao()
        p = self.precisao()
        return (1+b**2) * ((r* p)/(r+(b**2)*p))
    def f1_score(self):
        """Método que cálcula o F1-score a partir dos dados de saídas fornecidos.

        Returns:
            float: F1-score -> 2 * revocação * precisão/(revocação+precisão)
        """
        return self.fb_score()
