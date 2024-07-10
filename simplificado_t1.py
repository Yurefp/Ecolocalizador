import numpy as np
import math as mt
import matplotlib.pyplot as plt
from gera_csv import deltaD

#--------------------------------------------------------------------------------------------------------------
# Variáveis auxiliares
cicl = 0                #conta o número de repeções do metedo de Newton-Raphson
[xant, yant] = [0,0]    
#--------------------------------------------------------------------------------------------------------------
#Dandos conhecidos
[c, l] = [4, 4]
[x1, y1] = [0,0]
[x2, y2] = [0,c]
[x3, y3] = [l,0]
s1 = [x1, y1]
s2 = [x2, y2]
s3 = [x3, y3]
#--------------------------------------------------------------------------------------------------------------
#Valores calculados através dos sensores, como não tenho os sensores simulei os dados recebidos por eles
e = [1.3, 3.7] 
[D12, D13] = deltaD(s1, s2, s3, e, 4)
#--------------------------------------------------------------------------------------------------------------
[x, y] = [l/2, c/2]

#==================================================================================================================

def Df1x(x, y):
    return (x-x1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (x-x2)/(mt.sqrt((x-x2)**2 + (y-y2)**2))

def Df1y(x, y):
    return (y-y1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (y-y2)/(mt.sqrt((x-x2)**2 + (y-y2)**2))

def Df2x(x, y):
    return (x-x1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (x-x3)/(mt.sqrt((x-x3)**2 + (y-y3)**2))

def Df2y(x, y):
    return (y-y1)/(mt.sqrt((x-x1)**2 + (y-y1)**2)) - (y-y3)/(mt.sqrt((x-x3)**2 + (y-y3)**2))

#===================================================================================================================

def c1(x, y, i):
    a=0
    b=0
    while(mt.sqrt((x-x1)**2 + (y-y1)**2) < 2**a):
        b = a
        a=a-1 
    if(abs(mt.sqrt((x-x1)**2 + (y-y1)**2) - 2**a) < abs(mt.sqrt((x-x1)**2 + (y-y1)**2) - 2**b)):
        if(i==x):
            c = (x-x1)*(2**a)
        else:
            c = (y-y1)*(2**a)
    else:
        if(i==x):
            c = (x-x1)*(2**b)
        else:
            c = (y-y1)*(2**b)

    return c

def df1x(x, y):

    a=0
    b=0
    while(mt.sqrt((x-x1)**2 + (y-y1)**2) < 2**a):
        b = a
        a=a-1 
    if(abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**a) < abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**b)):
        d = (x-x2)*(2**a)
    else:
        d = (x-x2)*(2**b)

    return c1(x,y,x)-d

def df1y(x, y):
    a=0
    b=0
    while(mt.sqrt((x-x2)**2 + (y-y2)**2) < 2**a):
        b = a
        a=a-1 
    if(abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**a) < abs(mt.sqrt((x-x2)**2 + (y-y2)**2) - 2**b)):
        d = (y-y2)*(2**a)
    else:
        d = (y-y2)*(2**b)

    return c1(x,y,y)-d

def df2x(x, y):
    a=0
    b=0
    while(mt.sqrt((x-x3)**2 + (y-y3)**2) < 2**a):
        b = a
        a=a-1 
    if(abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**a) < abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**b)):
        d = (x-x3)*(2**a)
    else:
        d = (x-x3)*(2**b)

    return c1(x,y,x)-d

def df2y(x, y):
    a=0
    b=0
    while(mt.sqrt((x-x3)**2 + (y-y3)**2) < 2**a):
        b = a
        a=a-1 
    if(abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**a) < abs(mt.sqrt((x-x3)**2 + (y-y3)**2) - 2**b)):
        d = (y-y3)*(2**a)
    else:
        d = (y-y3)*(2**b)

    return c1(x,y,y)-d

#===================================================================================================================

def f1(x, y):
    return mt.sqrt((x-x1)**2 + (y-y1)**2) - mt.sqrt((x-x2)**2 + (y-y2)**2) - 2.5876

def f2(x, y):
    return mt.sqrt((x-x1)**2 + (y-y1)**2) - mt.sqrt((x-x3)**2 + (y-y3)**2) + 0.6587

'''while(abs(x - xant) > 10**(-3) and abs(y - yant) > 2**(-3)):
    cicl = cicl+1'''
for i in range (0,10):
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
    print("J1.f=", vprox)'''
    print("Resultado parcial:", [x, y])
    print("============================================")


'''def funcao(t, x, y):
    i=0
    for i in range (0,t):
        J = np.array([[Df1x(x, y), Df1y(x, y)],[Df2x(x, y), Df2y(x, y)]])
        J1 = np.linalg.inv(J)
        f = [f1(x, y), f2(x, y)]
        vprox = J1.dot(f)

        [x, y] = [x, y]-vprox

    return x'''

print("Resultado final:", [x, y])
print("Ciclos:", cicl)

'''amostra = 1000
t=[]
p=[]
for i in range (0,amostra):
    t.append(i)

for i in range (0,amostra):
    p.append(funcao(t[i], x, y))

plt.plot(t, p)
plt.xlabel('Valores de X')
plt.ylabel('Valores de Y')
plt.title('Gráfico de Linha')
plt.grid(True)
plt.show()'''
