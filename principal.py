import numpy as np
import math as mt
import matplotlib.pyplot as plt

#===================================================================================================================
#VARIAVEIS
#===================================================================================================================
# Variáveis auxiliares
cicl = 0                #conta o número de repeções do metedo de Newton-Raphson
[xant, yant] = [0,0]    

#===================================================================================================================
#Dandos conhecidos
[l, c] = [4, 4]         #largura e comprimento da sala
s1 = [x1, y1] = [0,0]   #posição do sensor 1
s2 = [x2, y2] = [0,c]   #posição do sensor 2
s3 = [x3, y3] = [l,0]   #posição do sensor 3

#===================================================================================================================
e = [1.5, 3.2] 
[x, y] = [l/2, c/2]


#===================================================================================================================
#FUNÇÕES
#===================================================================================================================
#Função para gerar a diferença entre as distâncias
def deltaD(s1, s2, s3, e, cd):
    D1 = mt.sqrt((e[0] - s1[0])**2 + (e[1] - s1[1])**2)
    D2 = mt.sqrt((e[0] - s2[0])**2 + (e[1] - s2[1])**2)
    D3 = mt.sqrt((e[0] - s3[0])**2 + (e[1] - s3[1])**2)

    D12 = D1 - D2
    D13 = D1 - D3

    return [round(D12, cd), round(D13, cd)]

#Valores calculados através dos sensores, como não tenho os sensores simulei os dados recebidos por eles
[D12, D13] = deltaD(s1, s2, s3, e, 4)

#===================================================================================================================
#Funções encontradas
def f1(x, y):
    return mt.sqrt((x-x1)**2 + (y-y1)**2) - mt.sqrt((x-x2)**2 + (y-y2)**2) - D12

def f2(x, y):
    return mt.sqrt((x-x1)**2 + (y-y1)**2) - mt.sqrt((x-x3)**2 + (y-y3)**2) - D13

#==================================================================================================================
#Funções das derivadas parciais normais
def Df1x(x, y):
    return (x-x1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (x-x2)/(mt.sqrt((x-x2)**2 + (y-y2)**2))

def Df1y(x, y):
    return (y-y1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (y-y2)/(mt.sqrt((x-x2)**2 + (y-y2)**2))

def Df2x(x, y):
    return (x-x1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (x-x3)/(mt.sqrt((x-x3)**2 + (y-y3)**2))

def Df2y(x, y):
    return (y-y1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (y-y3)/(mt.sqrt((x-x3)**2 + (y-y3)**2))

#===================================================================================================================
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

#===================================================================================================================
#Função para Newton-Raphson normal
def fun_norm(t, x, y):      #Função normal
    i=0
    for i in range (0,t):
        J = np.array([[Df1x(x, y), Df1y(x, y)],[Df2x(x, y), Df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)

        [x, y] = [x, y]-vprox

    return [x, y]

#Função para Newton-Raphson normal arredondado
def fun_arr(t, x, y):       #Função arredondano
    i=0
    for i in range (0,t):
        J = np.array([[df1x(x, y), df1y(x, y)],[df2x(x, y), df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)

        [x, y] = [x, y]-vprox

    return [x, y]

#Gera gráfio dos resultados dos dois metodos de Newton-Raphson em função do número de repetições
'''amostra = 10
t=[]
for i in range (0,amostra):
    t.append(i)

p_norm=[]
p_arr=[]
emis=[]
for i in range (0,amostra):
    p_norm.append(fun_norm(t[i], x, y)[0])
    p_arr.append(fun_arr(t[i], x, y)[0])
    emis.append(e[0])

plt.plot(t, p_norm)
plt.plot(t, p_arr)
plt.plot(t, emis)
plt.xlabel('Número de Repetições')
plt.ylabel('Valores de Y')
plt.title('Y em função do nº de Repetições')
plt.legend(['Normal', 'Arredondado', 'Posição Exata'])
plt.grid(True)
plt.show()'''
#===================================================================================================================
#Função de quantidade de repetições para x% de precição

def fun_arr_perc(p, x, y):      #Função arredondando
    cicl=0
    while((x/e[0] < p or x/e[0] > (2-p)) and (y/e[1] < p or y/e[1] > p)):
        J = np.array([[df1x(x, y), df1y(x, y)],[df2x(x, y), df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)
        [x, y] = [x, y]-vprox

        cicl = cicl+1

    #print('ciclos:', cicl)

    return cicl

def fun_norm_perc(p, x, y):     #Função normal
    cicl=0
    while((x/e[0] < p or x/e[0] > (2-p)) and (y/e[1] < p or y/e[1] > p)):
        J = np.array([[Df1x(x, y), Df1y(x, y)],[Df2x(x, y), Df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)
        [x, y] = [x, y]-vprox

        cicl = cicl+1

    #print('ciclos:', cicl)

    return cicl

#fun_arr_perc(0.60, x, y)
#print(fun_arr_perc(0.95, x, y))

#Gera gráfio do número de repetições em função do percenteu dos resultados dos dois metodos de Newton-Raphson 
'''perc=[]
for i in range (0,100):
    perc.append(i)

perc_norm=[]
perc_arr=[]
for i in range (0,100):
    perc_norm.append(fun_norm_perc(perc[i]/100, x, y))
    perc_arr.append(fun_arr_perc(perc[i]/100, x, y))

plt.plot(perc, perc_norm)
plt.plot(perc, perc_arr)
plt.xlabel('Percetual')
plt.ylabel('Número de Repetições')
plt.title('Y em função do nº de Repetições')
plt.legend(['Normal', 'Arredondado'])
plt.grid(True)
plt.show()'''

#===================================================================================================================

#===================================================================================================================
while(abs(x - xant) > 10**(-3) and abs(y - yant) > 2**(-3)):
    cicl = cicl+1
#for i in range (0,10):
    J = np.array([[Df1x(x, y), Df1y(x, y)],[Df2x(x, y), Df2y(x, y)]])
    J1 = np.linalg.inv(J)
    f = [f1(x, y), f2(x, y)]
    vprox = J1.dot(f)

    [xant, yant] = [x, y]
    [x, y] = [x, y]-vprox

    '''print("============================================")
    print("Resultado parcial anterior:", [xant, yant])
    print("--------------------------------------------")
    print("J=\n", J)
    print("J1=\n", J1)
    print("f=", f)
    print("J1.f=", vprox)
    print("Resultado parcial:", [x, y])
    print("============================================")'''


'''print("Resultado final:", [x, y])
print("Ciclos:", cicl)'''

#===================================================================================================================
#Gera uma imagem da sala no plano com a posição dos sesnores, do emissor e da posição calculada do emissor
'''plt.scatter(e[0], e[1])
plt.scatter(x, y, s=10)
plt.scatter([x1, x2, x3], [y1, y2, y3], c='r')
plt.grid(True)
plt.show()'''


#===================================================================================================================
#Gera gráfio dos resultados dos dois metodos de Newton-Raphson em função do número de repetições
'''amostra = 100
t=[]
for i in range (0,amostra):
    t.append(i)

p_norm=[]
p_arr=[]
emis=[]
for i in range (0,amostra):
    p_norm.append(fun_norm(t[i], x, y)[1])
    p_arr.append(fun_arr(t[i], x, y)[1])
    emis.append(e[1])

plt.plot(t, p_norm)
plt.plot(t, p_arr)
plt.plot(t, emis)
plt.xlabel('Número de Repetições')
plt.ylabel('Valores de Y')
plt.title('Y em função do nº de Repetições')
plt.legend(['Normal', 'Arredondado', 'Posição Exata'])
plt.grid(True)
plt.show()'''
