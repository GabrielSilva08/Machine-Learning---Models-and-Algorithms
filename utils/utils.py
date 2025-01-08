import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from models.florestaaleatoria import FlorestaAleatoria

breastcancer = np.genfromtxt(r"..\datasets\breastcancer.csv", delimiter=",")
vehicle = np.genfromtxt(r"..\datasets\vehicle.csv", delimiter=",")
kc2 = np.genfromtxt(r"..\datasets\kc2.csv", delimiter=",")
concrete = np.genfromtxt(r"..\datasets\concrete.csv", delimiter=",")
vowel = np.genfromtxt(r"..\datasets\vowel.csv", delimiter=",")
californiabin = np.genfromtxt(r"..\datasets\californiabin.csv", delimiter=",")
quake = np.genfromtxt(r"..\datasets\quake.csv", delimiter=",")
penguins = np.genfromtxt(r"..\datasets\penguins.csv", delimiter=",")

def treine_teste_divida(X, y, /, *, train_size=0.8, random_state=-1):
    """
    -----
    Descrição:
    
    Função que realiza o embaralhamento e a divisão dos dados de entrada e saída em 2 conjuntos.
    O conjunto de treino, correspondente à train_size% dos dados;
    o conjunto de teste, correspondente ao restante.

    -----
    Parâmetros:

    treino_test_divida(X, y, /, *, train_size=0.8)

    X -> Dados de entrada
    y -> Dados de saída
    train_size -> Porcentagem dos dados dedicada ao treinamento. Por padrão, 80%.
    random_state -> Parâmetro que garante a reproducibilidade (ou não) da divisão. Por padrão, -1 (não reprodutível).

    -----
    Retorno:

    X_train, X_test, y_train, y_test

    Conjuntos de dados de entrada e saída já divididos e embaralhados.
    """
    K = y.shape[1]
    Xy = np.hstack([X, y])
    if random_state != -1:
        np.random.seed(random_state)
    np.random.shuffle(Xy)
    # Pegando train_size% da matriz Xy já embaralhada
    t = int(train_size*Xy.shape[0])
    Xy_train = Xy[0:t]
    Xy_test = Xy[t:]
    # Fazendo a semparação dos dados de entrada e saída
    X_train = Xy_train[:, 0:Xy_train.shape[1]-K]
    y_train = Xy_train[:, Xy_train.shape[1]-K:]
    X_test = Xy_test[:, 0:Xy_test.shape[1]-K]
    y_test = Xy_test[:, Xy_test.shape[1]-K:]
    
    return X_train, X_test, y_train, y_test

def treine_valide_teste_divida(X, y, /, *, train_size=0.6, random_state=42):
    """
    -----
    Descrição:
    
    Função que realiza o embaralhamento e a divisão dos dados de entrada e saída em 3 conjuntos.

    -----
    Parâmetros:

    treino_test_divida(X, y, /, *, train_size=0.6)

    X -> Dados de entrada
    y -> Dados de saída
    train_size -> Porcentagem dos dados dedicada ao treinamento.

    -----
    Retorno:

    X_train, X_val, X_test, y_train, y_val, y_test

    Conjuntos de dados de entrada e saída já divididos e embaralhados.
    """
    np.random.seed(random_state)
    K = y.shape[1]
    Xy = np.hstack([X, y])
    np.random.shuffle(Xy)
    # Pegando train_size% da matriz Xy já embaralhada
    t = int(train_size*Xy.shape[0])
    Xy_train = Xy[0:t]
    Xy_test = Xy[t:]
    # Fazendo a semparação dos dados de entrada e saída
    X_train = Xy_train[:, 0:Xy_train.shape[1]-K]
    y_train = Xy_train[:, Xy_train.shape[1]-K:]
    X_test = Xy_test[:, 0:Xy_test.shape[1]-K]
    y_test = Xy_test[:, Xy_test.shape[1]-K:]
    val_idx = X_test.shape[0]//2
    X_val = X_test[0:val_idx]
    y_val = y_test[0:val_idx]
    X_test = X_test[val_idx:]
    y_test = y_test[val_idx:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def kfold(Xy_train, k):
    """Método que realiza a divisão de Xy_train em k folds.

    Args:
        Xy_train (_type_): Conjunto dos dados de treinamento onde será feito a validação.
        k (int): Número de folds
    
    Returns:
        folds (list): Lista de folds de índices de 0 a k-1.
    """
    folds = []
    idxs = np.linspace(0, Xy_train.shape[0], num=k+1, dtype=int)
    for i in range(k):
        folds.append(Xy_train[idxs[i]:idxs[i+1]+1,:])
    return folds

def acc(y_real, y_pred, K=2, classe="global"):
    """Função que realiza a computação da média e o desvio padrão da acurácia entre y_real e y_pred.

    Args:
        y_real (_type_): Dados de saídas reais
        y_pred (_type_): Dados de saídas preditos
        k (int): Número de classes
        classe (_type_, optional): Classe a qual serão feitas as computações. Por padrão, None (cálculo sem distinção de classe).

    Returns:
        media (_type_): Média computada sob o vetor de acertos
        desvio (_type_): Desvio computado sob o vetor de acertos
    """
    if classe == "global":
        acertos = (y_real == y_pred).astype("int8")
        media = acertos.mean()
        desvio = acertos.std()
        return media, desvio
    elif classe == "local":
        total = np.empty(K, dtype="int8")
        acertos = []
        for k in range(K):
            total[k] = np.unique(y_real, return_counts=True)[1][k]
            i=0
            acertos.append(np.zeros(total[k]))
            for idx in range(y_real.shape[0]):
                if y_real[idx] == k:
                    if y_pred[idx] == k:
                        acertos[k][i] += 1
                    i += 1
        medias = []
        desvios = []
        for acerto in acertos:
            medias.append(np.mean(acerto))
            desvios.append(np.std(acerto))
        return medias, desvios

# Como não o defini por meio de uma classe, ele ficará aqui em utils. Mas, one_hot_encoding 
# se enquadra como uma função de pré-processamento, afinal, pré-processa os dados de saída multiclasse.

def one_hot_encoding(y, k):
    """Aplicação da técnica de pré-processamento one hot encoding.

    Args:
        y (_type_): Dados de saída com valores variando num range de classes [1, k]
        k (int): Número de classes de saída

    Returns:
        Y (_type_): Dados de saída codificados com k colunas, onde cada uma faz referência a saída da classe original.
    """
    n = y.shape[0]
    Y = np.zeros((n, k))
    for i in range(n):
        Y[i][int(y[i][0])-1] += 1
    return Y

def grid_search_SVM(search_space, Xy_train, k):
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

    for c in search_space["C"]:
        for g in search_space["gamma"]:
            # Avaliação de uma combinação de hiperparâmetros
            classificador_SVM = SVC(C=c, kernel="rbf", gamma=g, probability=False)
            folds = kfold(Xy_train, k)
            acuracias = []
            # k fold
            for f in range(k):
                valid_fold = folds.pop()
                train_fold = np.vstack(folds)
                classificador_SVM.fit(train_fold[:,:-1], train_fold[:,-1])
                y_pred = classificador_SVM.predict(valid_fold[:,:-1])
                acuracias.append(acc(valid_fold[:,-1], y_pred)[0])
                folds.insert(0, valid_fold)
            acuracia_media = np.array(acuracias).mean()
            # Salvando o resultado da melhor combinação
            if acuracia_media > melhor_acuracia_media:
                melhor_acuracia_media = acuracia_media
                melhores_hiperparametros = (c, g)
    return melhor_acuracia_media, melhores_hiperparametros

def grid_search_RF(search_space, Xy_train, k):
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

    for n in search_space["n_modelos_base"]:
        for p in search_space["profundidade_maxima"]:
            # Avaliação de uma combinação de hiperparâmetros
            classificador_RF = FlorestaAleatoria(n_modelos_base=n, profundidade_maxima=p, random_state=42)
            folds = kfold(Xy_train, k)
            acuracias = []
            # k fold
            for f in range(k):
                valid_fold = folds.pop()
                train_fold = np.vstack(folds)
                classificador_RF.ajuste(train_fold[:,:-1], train_fold[:,[-1]])
                y_pred = classificador_RF.prever(valid_fold[:,:-1])
                acuracias.append(acc(valid_fold[:,-1], y_pred)[0])
                folds.insert(0, valid_fold)
            acuracia_media = np.array(acuracias).mean()
            # Salvando o resultado da melhor combinação
            if acuracia_media > melhor_acuracia_media:
                melhor_acuracia_media = acuracia_media
                melhores_hiperparametros = (n, p)
    return melhor_acuracia_media, melhores_hiperparametros

def grid_search_RF_sklearn(search_space, Xy_train, k):
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

    for n in search_space["n_modelos_base"]:
        for p in search_space["profundidade_maxima"]:
            # Avaliação de uma combinação de hiperparâmetros
            classificador_RF = RandomForestClassifier(n_estimators=n, max_depth=p, bootstrap=True, n_jobs=-1, random_state=42)
            folds = kfold(Xy_train, k)
            acuracias = []
            # k fold
            for f in range(k):
                valid_fold = folds.pop()
                train_fold = np.vstack(folds)
                classificador_RF.fit(train_fold[:,:-1], train_fold[:,-1])
                y_pred = classificador_RF.predict(valid_fold[:,:-1])
                acuracias.append(acc(valid_fold[:,-1], y_pred)[0])
                folds.insert(0, valid_fold)
            acuracia_media = np.array(acuracias).mean()
            # Salvando o resultado da melhor combinação
            if acuracia_media > melhor_acuracia_media:
                melhor_acuracia_media = acuracia_media
                melhores_hiperparametros = (n, p)
    return melhor_acuracia_media, melhores_hiperparametros

def davies_bouldin(clusters):
    DB_index = 0
    db_k = 0
    k = len(clusters)
    m = np.zeros((k, clusters[0][0].size))
    for i, c in enumerate(clusters):
        m[i] = c.mean(axis=0)
    
    for i in range(k):
        for j in range(k):
            if i != j:
                di = dist(clusters[i], m[i]).mean()
                dj = dist(clusters[j], m[j]).mean()
                Dij = dist(m[i], m[j])
                db_k = max(db_k, (di+dj)/Dij)
        DB_index += db_k
    return DB_index/k
            
def dist(x, y):
    if x.ndim != 1:
        return np.sqrt(((y - x) ** 2).sum(axis = 1).reshape(-1, 1))
    return np.sqrt(((x - y) ** 2).sum())