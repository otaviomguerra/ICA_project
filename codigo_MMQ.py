#TC1 ICA - Deteccao de spam
#Autor: Otavio Augusto Mota Guerra - 371849
#Resolvendo pelo MMQ(metodo dos minimos quadrados)

# Importando as bibliotecas
import pandas as pd
import numpy as np
import statistics as st

# Importando o banco de dados
dataset = pd.read_csv("spambase.data")
X = dataset.iloc[:, :57].values
y = dataset.iloc[:, 57].values

Acertos_positivos = []
Acertos_negativos = []
Acertos_global = []
Numero_FP = []
Numero_FN = []

for i in range(0,100):                
    # Separando em conjunto de testes e conjunto de treino 80/20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = i)
    
    # Normalizando as variaveis de treino e teste
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #Calculando a matriz de pesos W
    y_train[y_train == 0] = -1        
    W = np.dot(np.linalg.pinv(X_train),y_train)
    
    #Obtendo os resultados dos pesos obtidos para o conjunto de testes
    
    y_pred = np.dot(W, X_test.transpose())
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = 0
          
    # Fazendo a matriz de confusao
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    '''TP = # True Positives, TN = # True Negatives,
     FP = # False Positives, FN = # False Negatives
    
    Acerto = TP / (TP + FP)
    '''      
    #Aplicando o metodo de avaliacao
    
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    
    Acerto_positivo = TP / (TP + FP)
    Acerto_negativo = TN / (TN + FN)
    
    Acertos_positivos.append(Acerto_positivo)
    Acertos_negativos.append(Acerto_negativo)
    Acerto_global = (TP + TN) / (FP + FN + TP + TN)
    Numero_FP.append(FP)
    Numero_FN.append(FN)
    Acertos_global.append(Acerto_global)
    
#Calculo das medias de acerto por classe
Media_positivos = sum(Acertos_positivos) / float(len(Acertos_positivos))
Media_negativos = sum(Acertos_negativos) / float(len(Acertos_negativos))
#Calculo do MAIOR taxa de acerto obtida por classe
Maior_positivos = max(Acertos_positivos)
Maior_negativos = max(Acertos_negativos)
#Calculo da MENOR taxa de acerto obtida por classe
Menor_positivos = min(Acertos_positivos)
Menor_negativos = min(Acertos_negativos)
#Calculo do DESVIO PADRAO da taxa de acerto por classe
Desvio_positivos = st.stdev(Acertos_positivos)
Desvio_negativos = st.stdev(Acertos_negativos)
#Calculo da media de Falso positivos e Falso Negativos
Media_FP = sum(Numero_FP) / float(len(Numero_FP))
Media_FN = sum(Numero_FN) / float(len(Numero_FN))

#
Media_global = sum(Acertos_global) / float(len(Acertos_global))