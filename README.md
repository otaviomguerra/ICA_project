# ICA_project
Essa pasta contém o projeto final da Disciplina de Inteligência Computacional Aplicada oferecida no curso de Engenharia de Computação UFC.

O projeto consiste em classificar emails como spam ou não spam, com base no conjuntos de dados públicos presente neste link:https://archive.ics.uci.edu/ml/datasets/spambase .

## Procedimento
Foram utilizados 2 algorítimos de classificação diferentes codificados em Python: Perceptron Simples(codigo_perceptron.py) e o Método dos Mínimos Quadrados(codigo_MMQ.py). Já que se trata de um problema linearmente separável.

Como um dos requerimentos do projeto, o conjunto de dados teria que ser separado na regra 80/20. 80% para treinamento e 20% para teste/validação 100 vezes de maneira aleatoria. Calculando-se então: média, maior, menor e desvio padrão valor de acerto por classe. Bem como, a média de Falso Positivos, Falso Negativos e média global de acertos.

### Bibliotecas utilizadas:
Sklearn,
Pandas,
Numpy e 
Statistics(para cálculo do desvio padrão).


