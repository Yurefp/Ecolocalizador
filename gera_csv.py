import numpy as np
import math as mt
import csv

from principal import *

#===============================================================================================================
#Função para gerar a diferença entre as distâncias
def deltaD(s1, s2, s3, e, cd):
    D1 = mt.sqrt((e[0] - s1[0])**2 + (e[1] - s1[1])**2)
    D2 = mt.sqrt((e[0] - s2[0])**2 + (e[1] - s2[1])**2)
    D3 = mt.sqrt((e[0] - s3[0])**2 + (e[1] - s3[1])**2)

    D12 = D1 - D2
    D13 = D1 - D3

    return [round(D12, cd), round(D13, cd)]

#===============================================================================================================
#Funções encontradas
def f1(x, y):
    return mt.sqrt((x-x1)**2 + (y-y1)**2) - mt.sqrt((x-x2)**2 + (y-y2)**2) - D12

def f2(x, y):
    return mt.sqrt((x-x1)**2 + (y-y1)**2) - mt.sqrt((x-x3)**2 + (y-y3)**2) - D13

#===============================================================================================================
#Funções das derivadas parciais normais
def Df1x(x, y):
    return (x-x1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (x-x2)/(mt.sqrt((x-x2)**2 + (y-y2)**2))

def Df1y(x, y):
    return (y-y1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (y-y2)/(mt.sqrt((x-x2)**2 + (y-y2)**2))

def Df2x(x, y):
    return (x-x1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (x-x3)/(mt.sqrt((x-x3)**2 + (y-y3)**2))

def Df2y(x, y):
    return (y-y1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (y-y3)/(mt.sqrt((x-x3)**2 + (y-y3)**2))

#===============================================================================================================
#Funções das derivadas parciais arredondadas
def c1(x, y, i):
    log = mt.log2(mt.sqrt((x-x1)**2 + (y-y1)**2))
    b = mt.floor(log)
    a = mt.ceil(log)
    if(abs(mt.sqrt((x-x1)**2 + (y-y1)**2) - 2**a) < abs(mt.sqrt((x-x1)**2 + (y-y1)**2) - 2**b)):
        if(i==x):
            c = (x-x1)*(2**(-a))
        else:
            c = (y-y1)*(2**(-a))
    else:
        if(i==x):
            c = (x-x1)*(2**(-b))
        else:
            c = (y-y1)*(2**(-b))

    return c

def df1x(x, y):
    log = mt.log2(mt.sqrt((x-x2)**2 + (y-y2)**2))
    b = mt.floor(log)
    a = mt.ceil(log)
    if(abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**a) < abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**b)):
        d = (x-x2)*(2**(-a))
    else:
        d = (x-x2)*(2**(-b))

    return c1(x,y,x)-d

def df1y(x, y):
    log = mt.log2(mt.sqrt((x-x2)**2 + (y-y2)**2))
    b = mt.floor(log)
    a = mt.ceil(log)
    if(abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**a) < abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**b)):
        d = (y-y2)*(2**(-a))
    else:
        d = (y-y2)*(2**(-b))

    return c1(x,y,y)-d

def df2x(x, y):
    log = mt.log2(mt.sqrt((x-x3)**2 + (y-y3)**2))
    b = mt.floor(log)
    a = mt.ceil(log)
    if(abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**a) < abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**b)):
        d = (x-x3)*(2**(-a))
    else:
        d = (x-x3)*(2**(-b))

    return c1(x,y,x)-d

def df2y(x, y):
    log = mt.log2(mt.sqrt((x-x3)**2 + (y-y3)**2))
    b = mt.floor(log)
    a = mt.ceil(log)
    if(abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**a) < abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**b)):
        d = (y-y3)*(2**(-a))
    else:
        d = (y-y3)*(2**(-b))

    return c1(x,y,y)-d

#===============================================================================================================
#Função para contar a quantidade de repetições em função da acurácia
def ac_norm_perc(a, x, y):      #Função normal
    cicl=0
    while((x/e[0] < a/100 or x/e[0] > (2-a/100)) and (y/e[1] < a/100 or y/e[1] > (2-a/100))):
        J = np.array([[Df1x(x, y), Df1y(x, y)],[Df2x(x, y), Df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)
        [x, y] = [x, y]-vprox

        cicl = cicl+1
    return cicl

def ac_arr_perc(a, x, y):       #Função arredondada
    cicl=0
    while((x/e[0] < a/100 or x/e[0] > (2-a/100)) and (y/e[1] < a/100 or y/e[1] > (2-a/100))):
        J = np.array([[df1x(x, y), df1y(x, y)],[df2x(x, y), df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)
        [x, y] = [x, y]-vprox

        cicl = cicl+1
    return cicl
#===============================================================================================================
#Função para contar a quantidade de repetições em função da precisão
def pr_norm_perc(p, x, y):      #Função normal
    cicl=0
    [xant, yant] = [0, 0]
    while(abs(x - xant) > 10**(-p) and abs(y - yant) > 10**(-p)):
        cicl = cicl+1
        J = np.array([[Df1x(x, y), Df1y(x, y)],[Df2x(x, y), Df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)

        [xant, yant] = [x, y]
        [x, y] = [x, y]-vprox
    return cicl  

def pr_arr_perc(p, x, y):       #Função arredondada
    cicl=0
    [xant, yant] = [0, 0]
    while(abs(x - xant) > 10**(-p) and abs(y - yant) > 10**(-p)):
        cicl = cicl+1
        J = np.array([[df1x(x, y), df1y(x, y)],[df2x(x, y), df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)

        [xant, yant] = [x, y]
        [x, y] = [x, y]-vprox
    return cicl
#===============================================================================================================
#Gera a tabela para analise
with open('tabela.csv', 'w', newline='') as csvfile:
    titulos = ['Comprimento', 'Largura', 'X0', 'Y0', 'Xe', 'Ye', 'Distancia', 'Precisao', 'R.P.N.', 'R.P.A.', 'R.P.N.-R.P.A.', 'Preciso', 'Acuracia', 'R.A.N.', 'R.A.A.', 'R.A.N.-R.A.A.', 'Acurado']
    writer = csv.DictWriter(csvfile, fieldnames=titulos)

    writer.writeheader()
    for i in range(1, 100):                                  #Quantidade de linhas da tabela 
        l = np.random.randint(1,500)                        #Gera um valor aleatório para a largura da sala
        c = np.random.randint(1,500)                        #Gera um valor aleatório para o comprimento da sala
        s1 = [x1, y1] = [0,0]                               #posição do sensor 1
        s2 = [x2, y2] = [0,c]                               #posição do sensor 2
        s3 = [x3, y3] = [l,0]                               #posição do sensor 3

        xe = np.random.randint(1,l*1000)/1000               #Gera um valor aleatório para a posição x do emissor
        ye = np.random.randint(1,c*1000)/1000               #Gera um valor aleatório para a posição y do emissor
        e = [xe, ye]

        [D12, D13] = deltaD(s1, s2, s3, e, 3)               #Diferença das distâncias

        xden = 2**np.random.randint(1,4)                    #Gera n
        yden = 2**np.random.randint(1,4)                    #Gera m
        x0 = l/xden                                         #Gera um valor aleatório para a posição x0 n vezes menor que a largura da sala
        y0 = c/yden                                         #Gera um valor aleatório para a posição y0 m vezes menor que o comprimento da sala
        [x, y] = [x0, y0]

        dist = round(mt.sqrt((xe-x0)**2 + (ye-y0)**2), 3)   #Calcula a distancia entre o chute inicial a posição real do emissor
        a = np.random.randint(1,100)                        #Gera um valor aletório para a acurácia
        p = np.random.randint(0,4)                          #Gera um valor aletório para a precisão

        pr_n = pr_norm_perc(p, x, y)                        
        pr_a = pr_arr_perc(p, x, y)
        p_na = pr_n - pr_a                                  #Calcula a diferença entre o número de repetições necessárias para a precição escolhida
        if(p_na>0):
            mp = 'Arr'
        elif(p_na<0):
            mp = 'Norm'
        else:
            mp = 'Igual'

        ac_n = ac_norm_perc(a, x, y)
        ac_a = ac_arr_perc(a, x, y)
        a_na = ac_n - ac_a                                  #Calcula a diferença entre o número de repetições necessárias para a acurácia escolhida
        if(a_na>0):
            ma = 'Arr'
        elif(p_na<0):
            ma = 'Norm'
        else:
            ma = 'Igual'

        #Gera uma nova linha para a tabela
        writer.writerow({'Comprimento':c, 'Largura':l, 'X0':x0, 'Y0':y0, 'Xe':xe, 'Ye':ye, 'Distancia':dist, 'Precisao':(10**(-p)), 'R.P.N.':pr_n, 'R.P.A.':pr_a, 'R.P.N.-R.P.A.':p_na, 'Preciso':mp, 'Acuracia':a, 'R.A.N.':ac_n, 'R.A.A.':ac_a, 'R.A.N.-R.A.A.':a_na, 'Acurado':ma})






