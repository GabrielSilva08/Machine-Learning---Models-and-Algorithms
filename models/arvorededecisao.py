import numpy as np

class ArvoreDeDecisao:
    def __init__(self, indice_de_impureza="gini", impureza_minima=0, profundidade_maxima=None):
        """Método construtor do modelo árvore de decisão.

        Args:
            indice_de_impureza (str, optional): Indicativo de qual função custo utilizar. Por padrão, "gini".
                                                Valores aceitos -> ["gini", "entropia"]
            profundidade_maxima (int, optional): Indicativo do quão profunda a árvore pode no máximo ser. Por padrão, None (sem limites).
        """
        self.indice_de_impureza = indice_de_impureza
        # Objeto que permite visualizar a árvore treinada.
        self.visualizador = None
        # No raíz da árvore
        self.raiz = None
        self.impureza_minima = impureza_minima
        self.profundidade_maxima = profundidade_maxima
        
    def ajuste(self, X_train, y_train):
        """Método que realiza o treinamento do modelo.

        Args:
            X_train (_type_): Dados de treinamento de entrada
            y_train (_type_): Dados de treinamento de saída
        """
        if self.profundidade_maxima == None:
            self.profundidade_maxima = 1000000000000 # MAX
        Xy = np.hstack([X_train, y_train])
        # Cálculo do índice de impureza do nó atual (pai)
        I = self.funcao_custo(Xy[:, [-1]], self.indice_de_impureza)
        # Cálculo do melhor limiar a se fazer a divisão, sendo esse, apresentando impureza de partição mínima
        limiar = self.melhor_limiar(Xy, self.indice_de_impureza)
        # Salvar nó
        self.raiz = No(limiar["d"], limiar["t"], self.indice_de_impureza, I, Xy, 0)
        # Inicialização da Árvore de visualização
        self.visualizador = ArvoreVisualizador(self.raiz)
        # Chamada recursiva pros filhos
        self.treinamento(limiar["no_esquerdo"], "esquerdo", self.raiz)
        self.treinamento(limiar["no_direito"], "direito", self.raiz)
    
    def treinamento(self, Xy, paridade, no):
        """Método auxiliar que permite o treinamento recursivo de cada nó interno até 
           chegar no grau de pureza máxima (I = 0) ou antes, caso o parâmetro de 
           profundidade máxima tenha sido passado.

        Args:
            Xy (_type_): Dados do nó
            paridade (str): Relação de paridade com o nó pai (filho esquerdo ou filho direito)
            no (No): Nó pai
        """
        # Cálculo do índice de impureza do nó atual
        impureza = self.funcao_custo(Xy[:, [-1]], self.indice_de_impureza)
        # Cálculo da profundidade do nó atual
        profundidade = no.profundidade_no()+1
        # Verificação se o nó atual será folha (pureza máxima) ou se fora atingida a profundidade máxima
        if impureza <= self.impureza_minima or profundidade >= self.profundidade_maxima:
            if paridade == "esquerdo":
                no.esquerdo = No(0, 0, self.indice_de_impureza, impureza, Xy, True, profundidade)
            else:
                no.direito = No(0, 0, self.indice_de_impureza, impureza, Xy, True, profundidade)
            return
        # Cálculo do melhor limiar a se fazer a divisão, sendo esse apresentando impureza de partição mínima
        limiar = self.melhor_limiar(Xy, self.indice_de_impureza)

        

        if paridade == "esquerdo":
            # Filho esquerdo


            # Verificação se o filho filho esquerdo é o mesmo que o filho do pai
            if limiar["no_esquerdo"].shape[0] == 0 or (Xy.shape == limiar["no_esquerdo"].shape and np.all(Xy == limiar["no_esquerdo"])):
                no.esquerdo.esquerdo = None
                if limiar["no_direito"].shape[0] == 0 or (Xy.shape == limiar["no_direito"].shape and np.all(Xy == limiar["no_direito"])):
                    no.esquerdo.direito = None
                    no.esquerdo = No(0, 0, self.indice_de_impureza, impureza, Xy, True, profundidade)
                    return
                else:
                    no.esquerdo.direito = No(None, None, None, None, None)
                    # Chamada recursiva pros filhos do nó caso direito
                    self.treinamento(limiar["no_direito"], "direito", no.esquerdo)


            # Verificação se o filho filho direito é o mesmo que o filho do pai
            if limiar["no_direito"].shape[0] == 0 or (Xy.shape == limiar["no_direito"].shape and np.all(Xy == limiar["no_direito"])):
                no.esquerdo.direito = None
                if Xy.shape == limiar["no_esquerdo"].shape and np.all(Xy == limiar["no_esquerdo"]):
                    no.esquerdo.esquerdo = None
                    no.esquerdo = No(0, 0, self.indice_de_impureza, impureza, Xy, True, profundidade)
                    return
                else:
                    no.esquerdo.esquerdo = No(None, None, None, None, None)
                    # Chamada recursiva pros filhos do nó caso direito
                    self.treinamento(limiar["no_esquerdo"], "esquerdo", no.esquerdo)



            no.esquerdo = No(limiar["d"], limiar["t"], self.indice_de_impureza, impureza, Xy, False, profundidade)
            no.esquerdo.esquerdo = No(None, None, None, None, None)
            no.esquerdo.direito = No(None, None, None, None, None)
            # Chamada recursiva pros filhos do nó caso esquerdo
            self.treinamento(limiar["no_esquerdo"], "esquerdo", no.esquerdo)
            self.treinamento(limiar["no_direito"], "direito", no.esquerdo)
        else:
            # Filho direito

            # Verificação se o filho filho esquerdo é o mesmo que o filho do pai
            if limiar["no_esquerdo"].shape[0] == 0 or (Xy.shape == limiar["no_esquerdo"].shape and np.all(Xy == limiar["no_esquerdo"])):
                no.direito.esquerdo = None
                if limiar["no_direito"].shape[0] == 0 or (Xy.shape == limiar["no_direito"].shape and np.all(Xy == limiar["no_direito"])):
                    no.direito.direito = None
                    no.direito = No(0, 0, self.indice_de_impureza, impureza, Xy, True, profundidade)
                    return
                else:
                    no.direito.direito = No(None, None, None, None, None)
                    # Chamada recursiva pros filhos do nó caso direito
                    self.treinamento(limiar["no_direito"], "direito", no.direito)


            # Verificação se o filho filho direito é o mesmo que o filho do pai
            if limiar["no_direito"].shape[0] == 0 or (Xy.shape == limiar["no_direito"].shape and np.all(Xy == limiar["no_direito"])):
                no.direito.direito = None
                if limiar["no_esquerdo"].shape[0] == 0 or (Xy.shape == limiar["no_esquerdo"].shape and np.all(Xy == limiar["no_esquerdo"])):
                    no.direito.esquerdo = None
                    no.direito = No(0, 0, self.indice_de_impureza, impureza, Xy, True, profundidade)
                    return
                else:
                    no.direito.esquerdo = No(None, None, None, None, None)
                    # Chamada recursiva pros filhos do nó caso direito
                    self.treinamento(limiar["no_esquerdo"], "esquerdo", no.direito)




            no.direito = No(limiar["d"], limiar["t"], self.indice_de_impureza, impureza, Xy, False, profundidade)
            no.direito.esquerdo = No(None, None, None, None, None)
            no.direito.direito = No(None, None, None, None, None)
            # Chamada recursiva pros filhos do nó caso direito
            self.treinamento(limiar["no_esquerdo"], "esquerdo", no.direito)
            self.treinamento(limiar["no_direito"], "direito", no.direito)

    def prever(self, X_test):
        """Método que realiza a predição para novos padrões

        Args:
            X_test (_type_): Padrões que se desejam realizar novas predições

        Returns:
            y_pred (_type_): Vetor com os valores de saída preditos.
        """
        if X_test.ndim == 1:
            X_test = X_test[np.newaxis, :]
        y_pred = np.empty(X_test.shape[0]).reshape(-1, 1)
        for i, x in enumerate(X_test):
            no = self.raiz
            while no.ehfolha() == False:
                if(x[no.d] <= no.t):
                    no = no.esquerdo
                else:
                    no = no.direito
            y_pred[i] = no.classe()
        return y_pred
    
    def prever_proba(self, X_test):
        """Método que realiza a predição para novos padrões e retorna
           a proporção referente ao nó folha

        Args:
            X_test (_type_): Padrões que se desejam realizar novas predições

        Returns:
            y_proba (_type_): Vetor de probabilidades de saída preditos.
        """
        if X_test.ndim == 1:
            X_test = X_test[np.newaxis, :]
        y_proba = np.empty(X_test.shape[0]).reshape(-1, 1)
        for i, x in enumerate(X_test):
            no = self.raiz
            while no.ehfolha() == False:
                if(x[no.d] <= no.t):
                    no = no.esquerdo
                else:
                    no = no.direito
            y_proba[i] = no.classe_proba()
        return y_proba
                
    def visualizar(self):
        """Método que imprime a árvore de decisão já treinada.

        Returns:
            _type_: Visualização da Árvore de decisão
        """
        return self.visualizador.visualizar(self.raiz)
    
    def visualizar_simplificado(self):
        """Método que imprime a árvore de decisão já treinada de uma maneira simplificado.

        Returns:
            _type_: Visualização da Árvore de decisão
        """
        return self.visualizador.visualizar_simplificado(self.raiz)
    
    def visualizador_tree(self):
        """Método que imprime a árvore de decisão já treinada,

        Returns:
            _type_: Visualização da Árvore de decisão
        """
        return self.visualizador

    def melhor_limiar(self, Xy, indice):
        """Método auxiliar que realiza o cálculo do melhor limiar (d, t) por força bruta.
        Verificando cada candidato à atributo d com cada candidato a limiar t.

        Args:
            Xy (_type_): Dados do nó a se computar o limiar
            indice (str): Índice de impureza utilizado

        Returns:
            limiar (dict): Dicionário contendo informações do limiar computado
        """
        # Cálculo do limiar (d, t) que gera a menor impureza da partição.
        D = Xy.shape[1] - 1
        N = Xy.shape[0]
        # Melhor limiar
        limiar = {}
        limiar["pureza_da_particao"] = 2
        limiar["d"] = None
        limiar["t"] = None
        # Pra cada atributo
        for d in range(D):
            # Pra cada limiar (sendo esse um valor presente nos dados)
            for i in range(N):
                t = Xy[i, d]
                pureza, no_v, no_f = self.calc_pureza(Xy, d, t, indice)
                # Salva o limiar de menor impureza de partição
                if pureza < limiar["pureza_da_particao"]:
                    limiar["pureza_da_particao"] = pureza
                    limiar["d"] = d
                    limiar["t"] = t
                    limiar["no_esquerdo"] = no_v
                    limiar["no_direito"] = no_f
        return limiar

    def calc_pureza(self, Xy, d, t , indice):
        """Método auxiliar que calcula a pureza do nó de dados Xy e limiar (d, t).

        Args:
            Xy (_type_): Dados do nó a se computar o limiar
            d (int): Atributo
            t (float): Limiar
            indice (str): Índice de impureza utilizado

        Returns:
            it (float): Pureza total
            Xy_V (_type_): Dados do nó filho esquerdo
            Xy_F (_type_): Dados do nó filho direito
        """
        Xy_V = Xy[self.filter_by_condition(Xy, d, t, True)]
        Xy_F = Xy[self.filter_by_condition(Xy, d, t, False)]
        iV, nV = self.funcao_custo(Xy_V[:, [-1]], self.indice_de_impureza), Xy_V.shape[0]
        iF, nF = self.funcao_custo(Xy_F[:, [-1]], self.indice_de_impureza), Xy_F.shape[0]
        it = (nV * iV + nF * iF)/(nV+nF)
        return it, Xy_V, Xy_F
    
    def filter_by_condition(self, Xy, d, t, condition_state):
        """Método auxiliar que realiza a filtração dos dados do nó Xy que atendem ou não determinado limiar (d <= t).

        Args:
            Xy (_type_): Dados do nó
            d (int): Atributo
            t (float): Limiar
            condition_state (bool): Estado de qual limiar se deseja filtrar, (d <= t) ou (d > t)

        Returns:
            _type_: Indices de Xy filtrado pelo limiar
        """
        if condition_state == True:
            return [True if Xy[i, d] <= t else False for i in range(Xy.shape[0])]
        return [True if Xy[i, d] > t else False for i in range(Xy.shape[0])]
    
    def funcao_custo(self, y, indice):
        """Método auxiliar que realiza o computo de fato da pureza.

        Args:
            y (_type_): Vetor de categorias de saída
            indice (str): Índice de pureza utilizado

        Raises:
            Exception: Lançado quando o índice de pureza não fora especificado

        Returns:
            float: pureza
        """
        if y.size == 0:
            return 0
        classes, contagem = np.unique(y, return_counts=True)
        if classes.size == 1:
            if classes[0] == 0:
                n0 = contagem[0]
                n1 = 0
            else:
                n0 = 0
                n1 = contagem[0]
        else:
            n0, n1 = contagem[0], contagem[1]
        n = n0 + n1
        if indice == "gini":
            g = 1 - (n0/n) ** 2 - (n1/n) ** 2
            return g
        elif indice == "entropia":
            if n0 == 0 or n1 == 0:
                return 0
            h = (-1) * (((n0/n) * np.log2(n0/n)) + ((n1/n) * np.log2(n1/n)))
            return h
        else:
            raise Exception("Índice de impureza não especificado.")

class ArvoreVisualizador:
    def __init__(self, raiz):
        """Método construtor do visualizador de uma árvore de decisão.

        Args:
            raiz (No): Nó raiz da árvore.
        """
        self.raiz = raiz
    def percusopreordem(self, no="raiz"):
        """Método que percorre os nós presentes da árvore (imprimindo-os) num percuso em pré-ordem.

        Args:
            no (str): Indicativo de que o percuso irá começar pela raiz.
        """
        if no == "raiz":
            no = self.raiz
        if no != None:
            print(no)
            self.percusopreordem(no.esquerdo)
            self.percusopreordem(no.direito)
    def ordem_dos_nos(self, no):
        """Método que retorna uma lista com os nós visitados numa busca em largura (BFS).

        Args:
            no (No): Nó de partida da busca.

        Returns:
            ordem_visualizacao (list): Lista de nós visitados numa busca em largura.
        """
        fila = []
        ordem_visualizacao = []
        ordem_visualizacao.append(no)
        fila.append(no.esquerdo)
        fila.append(no.direito)
        while len(fila) != 0:
            ordem_visualizacao.append(fila[0])
            if fila[0].esquerdo != None:
                fila.append(fila[0].esquerdo)
            if fila[0].direito != None:
                fila.append(fila[0].direito)
            fila.pop(0)
        return ordem_visualizacao
    
    def visualizar(self, no, nivel=0):
        """Método que permite a visualização da árvore de decisão de uma maneira tabular e completa.
        Cada nível de indentação indica a profundidade do nó na árvore.

        Args:
            no (No): No que será primeiro visualizado.
            nivel (int, optional): Parâmetro indicador de o quão indentado estará o "no". Como a visualização é feita da raíz, 0.
        """
        if no != None:
            print(no.printNo(nivel))
            self.visualizar(no.esquerdo, nivel+1)
            self.visualizar(no.direito, nivel+1)
    
    def visualizar_simplificado(self, no, nivel=0):
        """Método que permite a visualização da árvore de decisão de uma maneira tabular e simples, mostrando somente o limiar/classe do nó.
        Cada nível de indentação indica a profundidade do nó na árvore.

        Args:
            no (No): No que será primeiro visualizado.
            nivel (int, optional): Parâmetro indicador de o quão indentado estará o "no". Como a visualização é feita da raíz, 0.
        """
        if no != None:
            print(no.printNoSimplificado(nivel))
            self.visualizar_simplificado(no.esquerdo, nivel+1)
            self.visualizar_simplificado(no.direito, nivel+1)

    def calc_altura(self, no):
        """Método que calcula a altura de "no".

        Args:
            no (No): No que se deseja computar a altura.

        Returns:
            int: Altura de "no"
        """
        if no.ehfolha():
            return 0
        return 1+max(self.calc_altura(no.esquerdo), self.calc_altura(no.direito))
        
class No:
    def __init__(self, atributo, limiar, nome_indice, impureza, Xy, folha=False, profundidade=0):
        """Método construtor de um nó presente na árvore de decisão.

        Args:
            atributo (int): Atributo presente na condicional (se tiver);
            limiar (_type_): Limiar presente na condicional (se tiver);
            nome_indice (_type_): Nome do índice de impureza adotado;
            impureza (_type_): Grau de impureza do índice;
            Xy (_type_): Dados do nó;
            folha (bool, optional): Informativo se o nó é folha ou não. Defaults to False.
            profundidade (int): Profundidade do nó
        """
        self.d = atributo
        self.t = limiar
        self.nome_indice = nome_indice
        self.impureza = impureza
        self.values = Xy
        self.esquerdo = None
        self.direito = None
        self.folha = folha
        self.profundidade = profundidade

    def __str__(self):
        """Método especial que encapsula o formato de sua impressão caso passado na função "print()".
        """
        classes, contagem = np.unique(self.values[:, [-1]], return_counts=True)
        if classes.size == 1:
            if classes[0] == 0:
                n0 = contagem[0]
                n1 = 0
            else:
                n0 = 0
                n1 = contagem[0]
        else:
            n0, n1 = contagem[0], contagem[1]
        c = classes[np.argmax(contagem)]
        if self.folha == False:
            return f"""--------------------
              |                  |
              |   d{self.d} <= {self.t}      |
              |   {self.nome_indice} = {self.impureza:.3f}   |
              |   N = {n0+n1}         |
              |   [{n0}, {n1}]         |
              |   Classe = {c}     |
              |                  |
              --------------------
            """
        return f"""--------------------
              |                  |
              |   Classe = {c}     |
              |   {self.nome_indice} = {self.impureza:.3f}   |
              |   N = {n0+n1}          |
              |   [{n0}, {n1}]         |
              |                  |
              --------------------
            """
    
    def printNo(self, nivel):
        """Método que imprime num formato completo, as informações do nó.

        Args:
            nivel (int): Nível de indentação (referente a profundidade) que esse será escrito.

        Returns:
            str: Informações do nó (limiar, indíce de pureza adotado, grau de impureza, número de dados por classe, classe majoritária).
        """
        classes, contagem = np.unique(self.values[:, [-1]], return_counts=True)
        if classes.size == 1:
            if classes[0] == 0:
                n0 = contagem[0]
                n1 = 0
            else:
                n0 = 0
                n1 = contagem[0]
        else:
            n0, n1 = contagem[0], contagem[1]
        c = classes[np.argmax(contagem)]
        tab = "\t" * nivel
        if self.folha == False:
            return f"""{tab}              --------------------
              {tab}|                  |
              {tab}|   d{self.d} <= {self.t}      |
              {tab}|   {self.nome_indice} = {self.impureza:.3f}   |
              {tab}|   N = {n0+n1}         |
              {tab}|   [{n0}, {n1}]         |
              {tab}|   Classe = {int(c)}     |
              {tab}|                  |
              {tab}--------------------
            """
        return f"""{tab}        --------------------
              {tab}|                  |
              {tab}|   Classe = {int(c)}     |
              {tab}|   {self.nome_indice} = {self.impureza:.3f}   |
              {tab}|   N = {n0+n1}          |
              {tab}|   [{n0}, {n1}]         |
              {tab}|                  |
              {tab}--------------------
            """
    
    def printNoSimplificado(self, nivel):
        """Método que imprime num formato simplificado, as informações do nó (limiar/classe).

        Args:
            nivel (int): Nível de indentação (referente a profundidade) que esse será escrito.

        Returns:
            str: Informações simplificadas do nó
        """
        classes, contagem = np.unique(self.values[:, [-1]], return_counts=True)
        if classes.size == 1:
            if classes[0] == 0:
                n0 = contagem[0]
                n1 = 0
            else:
                n0 = 0
                n1 = contagem[0]
        else:
            n0, n1 = contagem[0], contagem[1]
        c = classes[np.argmax(contagem)]
        tab = "\t" * nivel
        if self.folha == False:
            return f"{tab}d{self.d} <= {self.t}"
        return f"{tab}Classe = {int(c)}"
    
    def ehfolha(self):
        """Método informa se o nó em questão é folha ou não.

        Returns:
            bool: True (é folha) ou False (não é folha).
        """
        return self.folha
    
    def classe(self):
        """Método que retorna a classe do nó.

        Returns:
            c (int): Categoria do nó.
        """
        classes, contagem = np.unique(self.values[:, [-1]], return_counts=True)
        if classes.size == 1:
            if classes[0] == 0:
                n0 = contagem[0]
                n1 = 0
            else:
                n0 = 0
                n1 = contagem[0]
        else:
            n0, n1 = contagem[0], contagem[1]
        c = classes[np.argmax(contagem)]
        return int(c)
    
    def classe_proba(self):
        """Método que retorna a proporção refente à classe positiva (1) do nó.

        Returns:
            c (int): Categoria do nó.
        """
        classes, contagem = np.unique(self.values[:, [-1]], return_counts=True)
        if classes.size == 1:
            if classes[0] == 0:
                return 0
            else:
                return 1
        else:
            n0, n1 = contagem[0], contagem[1]
        return n1/(n1+n0)
    
    def profundidade_no(self):
        """Método que retorna a profundidade do nó

        Returns:
            profundidade (int): Profundidade do nó
        """
        return self.profundidade