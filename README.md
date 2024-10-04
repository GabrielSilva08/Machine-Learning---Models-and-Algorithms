# Machine Learning - Models and Algorithms

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg) ![Numpy](https://img.shields.io/badge/numpy-v1.26.4-4287f5.svg) ![Matplotlib](https://img.shields.io/badge/matplotlib-v3.7.3-important.svg)

Projeto feito para entender e praticar alguns modelos e algoritmos de aprendizado de máquinas.

## Modelos

Pretendo nesse projeto implementar o códigos dos seguintes modelos com respeito as tarefas de aprendizagem de máquina que visam resolver:

- Aprendizado Supervisionado 
    - Regressão
        - Regressão Linear (em andamento) - 🚧
        - Regressão Polinomial (em andamento) - 🚧
        - Naive Bayes Gaussiano (em andamento) - 🚧
        - K-NN (em andamento) - 🚧
        - Árvores de Decisão (em andamento) - 🚧
        - MLP (em andamento) - 🚧
        - SVM (em andamento) - 🚧
        - Random Forest (em andamento) - 🚧
    - Classificação Binária
        - Regressão Logística (em andamento) - 🚧
        - Análise de Discriminante Gaussiano (em andamento) - 🚧
        - Naive Bayes Gaussiano (em andamento) - 🚧
        - K-NN (em andamento) - 🚧
        - Árvores de Decisão (em andamento) - 🚧
        - MLP (em andamento) - 🚧
        - SVM (em andamento) - 🚧
        - Random Forest (em andamento) - 🚧
    - Classificação Multiclasse
        - Regressão Logística (em andamento) - 🚧
        - Análise de Discriminante Gaussiano (em andamento) - 🚧
        - Naive Bayes Gaussiano (em andamento) - 🚧
        - K-NN (em andamento) - 🚧
        - Árvores de Decisão (em andamento) - 🚧
        - MLP (em andamento) - 🚧
        - SVM (em andamento) - 🚧
        - Random Forest (em andamento) - 🚧
- Aprendizado Não-Supervisionado
    - Agrupamento (clustering)
        - K-means (em andamento) - 🚧
        - DBSCAN (em andamento) - 🚧
    - Redução de Dimensionalindade
        - PCA (em andamento) - 🚧

No decorrer desse projeto, pretendo utilizar somente as seguintes bibliotecas:

* `numpy`: Para poder realizar as manipulações necessárias com os vetores e matrizes. Será utilizado majoritariarmente para construir os algoritmos e os modelos.
* `matplotlib`: Para poder visualizar diferentes tipos de gráficos gerados pelos modelos. Também será utilizado para realização de análises exploratórias dos dados trabalhados em cada situação.

```bash
pip install numpy=1.26.4
pip install matplotlib=3.7.3
```

## Datasets

Escolhi trabalhar com alguns datasets que já havia explorado previamente nos meus estudos. Sendo eles:

- [california](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset): Dataset organizado em 20.640 amostras e 8 atributos. O dataset é utilizado na predição da mediana de preços de casas dos distritos da Califórnia na década de 90. A saída é um valor real, configurando uma **tarefa de regressão**.
- [breastcancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic): Dataset organizado em 569 instâncias e 30 atributos. O dataset é utilizado para a predição do diagnóstico de uma paciente, informando-a se ela possui ou não câncer de mama. A saída é um valor binário, 0 para a classe negativa e 1 para a classe positiva, configurando uma **tarefa de classificação binária**.
- [vehicle](https://www.openml.org/search?type=data&sort=runs&id=54&status=active): Dataset organizado em 946 padrões e 18 atributos. O dataset é utilizado para, a partir dos dados de silhuetas do veículo, classificar se esse é bus, opel, saab ou van. Sendo assim, configura-se uma **tarefa de classificação multiclasse** com 4 classes.
- [kc2](https://k8sapi.openml.org/d/1063): Dataset subamostrado de 21 features. Os dados são referentes as características presentes nos códigos-fontes para processamento de dados na NASA. A saída é um indicador de ausência (0) ou presença (1) de defeitos. Sendo assim, configura-se uma **tarefa de classificação binária**.
- [concrete](https://www.openml.org/search?type=data&sort=runs&id=4353&status=active): Dataset com 1030 instâncias e 8 atributos. Os atributos dizem respeito a diferentes tipos de concreto para construção. A saída é um valor real indicando a resistência à compressão do concreto. Sendo assim, uma **tarefa de regressão**.
- [vowel](https://www.openml.org/search?type=data&sort=runs&id=307&status=active): Dataset com 990 padrões e 10 atributos. Os atributos são elementos presentes nas falas de britânicos. A saída corresponde à um fonema de vogal, tendo 11 possibilidades. Um problema que configura-se a uma **tarefa de classificação multiclasse** com 11 classes.
- californiabin: Dataset similar ao dataset california, mas a saída foi convertida para um formato onde será 0 se for abaixo da mediana dos preços, 1 caso contrário. Tornando-se assim, uma **tarefa de classificação binária**.
- [quake](https://www.openml.org/search?type=data&sort=runs&id=772&status=active): Dataset composto por 2178 padrões e 2 colunas, onde essas referem-se as coordenadas de locais em que foram registrados terremotos. Nossa abordagem para esse problema pode ser de realizar uma clusterização de padrões similares, sendo assim, uma **tarefa de agrupamento**.
- [penguins](https://allisonhorst.github.io/palmerpenguins/): Dataset composto de 344 amostras e 4 atributos referentes às medidas anatômicos de pinguins da Antártida, onde esses pertencem a uma espécie de um total de 3. Podendo configurar um problema de classificação multiclasse, mas, tentarei trabalhar com a questão visando observar os atributos mais importantes, ou seja, uma **tarefa de redução de dimensionalidade**.

## Notebooks e Utils

Dentro de cada notebook estarei explorando cada modelo com os respectivos datasets selecionados. Estarei trabalhando com diversas métricas de avaliação bem como realizando comparações com os modelos já existentes da biblioteca `Sklearn`. O diretório `utils` tem função de apenas comportar classes e métodos auxiliares que poderão aparecer no treinamento dos modelos. Ex: K-fold, train_test_split, StandardScaler...

## Considerações

Estou aberto a qualquer comentário de melhoria ou sugestão. Meu objetivo é conseguir melhorar ainda mais o meu entendimento à respeito dessa imensa área que é o aprendizado de máquina.

## Referências de estudo

Bibliografia que estarei utilizando para me basear na escrita dos modelos e seu entendimento.

* DEISENROTH, M. et al. Mathematics for machine learning. Cambridge University Press, 2019 (https://mml-book.github.io/book/mml-book.pdf)

* MURPHY, K. Probabilistic Machine Learning: An Introduction. MIT Press, 2021 (https://github.com/probml/pml-book/releases/latest/download/book1.pdf)

* BISHOP, C. Pattern recognition and machine learning. Springer, 2006 (https://microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
