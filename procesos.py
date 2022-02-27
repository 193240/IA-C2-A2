import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from p1 import *
from numpy.linalg import norm



def randomsW():
    w= []
    for i in range(0,3):
        w.append(np.random.uniform(0,1))
    return w

def etaR():
    return np.random.uniform(0, 1)

def umbralR():
    return np.random.uniform(0,1)

def inicioR():
    X = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    Y = [0, 0, 0, 1]
    #X = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    #Y = [1, 1, 0, 1]
    r =[]
    for i in range(0, 5):
        Wcero = randomsW()
        eta = etaR()
        umbral = umbralR()
        iteracion = entrenamiento(Wcero, eta, umbral, X, Y)
        #grafIteracion(iteracion)
        r.append(iteracion)

    #for j in r:
        #print(j)

    graficacionEtas(r)
    peso_final(r)

def inicio():
    X = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    Y = [0, 0, 0, 1]
    Wcero = [0.854, 0.327, 0.558]
    eta = 0.4
    umbral = 0.5
    r = entrenamiento(Wcero, eta, umbral, X, Y)
    grafIteracion(r)
    t = r.pop()
    print(t["wk"])

def entrenamiento(Wcero,eta, umbral, X, Y):
    iteracion=0
    l = []
    while True:
        Wk = []
        if (iteracion == 0):
            Wk = Wcero
        else:
            Wk = wk.copy()

        Uk = calcularU(X, Wk)
        yc = yCalculada(Y, Uk)
        error = calcularError(Y, yc)
        ex = multiplicarEX(error, X)
        p = multiplicacionEta(eta, ex)
        wk = nuevoW(Wk, p)
        norma = np.linalg.norm(error)
        dic = {
            "iteracion":iteracion,
            "norma": norma,
            "wk": Wk,
            "eta": eta
        }
        l.append(dic)
        # iteramos la generacion
        iteracion += 1
        try:
            if norma < umbral:
                break
            #else:
                #print("norma no es menor a umbral: ","norma es: ",norma," umbral es: ",umbral)

        except:
            print("Ocurrio un ciclo")
    return l

def grafIteracion(lista):
    df = pd.DataFrame(lista)
    fig, ax = plt.subplots()
    ax.plot(df.index.values, df["norma"])
    plt.show()

def graficacionEtas(lista):
    l = []
    for i in lista:
        df = pd.DataFrame(i)
        mostrarGrafica = {
            "iteracion":df.index.values,
            "eta":df["eta"][0],
            "norma":df["norma"],
        }
        l.append(mostrarGrafica)

    df = pd.DataFrame(l)

    plt.figure(figsize=[6, 6])
    plt.plot(df["iteracion"][0], df["norma"][0], label="Eta numero1: "+str(df["eta"][0]))
    plt.plot(df["iteracion"][1], df["norma"][1], label="Eta numero2: "+str(df["eta"][1]))
    plt.plot(df["iteracion"][2], df["norma"][2], label="Eta numero3: "+str(df["eta"][2]))
    plt.plot(df["iteracion"][3], df["norma"][3], label="Eta numero4: "+str(df["eta"][3]))
    plt.plot(df["iteracion"][4], df["norma"][4], label="Eta numero5: "+str(df["eta"][4]))
    plt.xlabel('Generaciones')  # override the xlabel
    plt.ylabel('Fitness')  # override the ylabel
    plt.title('Norma error')  # override the title
    plt.legend()
    plt.show()


def peso_final(lista):
    for i in lista:
        dic= i.pop()
        print("eta:{0} pesosfinales {1} ".format(dic["eta"], dic["wk"]))

inicio()


''' print("____________________")
    print("Iteracion: ", iteracion)
    # aca calculamos Uk
    print("Wk", Wk)
    Uk = calcularU(X, Wk)
    print("Uk: ", Uk)
    # Aca hacemos yCalculada
    yc = yCalculada(Y, Uk)
    print("Y calculada: ", yc)
    # aca calculamos el error
    error = calcularError(Y, yc)
    print("error: ", error)
    # calculamos
    ex = multiplicarEX(error, X)
    print("ex: ", ex)
    # hacemos la multiplicacion de eta
    p = multiplicacionEta(eta, ex)
    print("multiplicacion eta: ", p)
    # se crea un nuevo w, y ya tenemos el valor de wk
    wk = nuevoW(Wk, p)
    print("nuevo wk: ", wk)
    # sacamos la norma de la w
    
    
    print("norma: ", norma)'''

    #if(np.array_equal(error,Y)):
        #guardariamos los resultados
     #   break
#df = pd.DataFrame(l)

#print("ULTIMO VALOR: ",df["wk"].iloc[-1])

#fig, ax = plt.subplots()
#ax.plot(df.index.values, df["norma"])
#plt.show()