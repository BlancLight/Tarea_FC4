# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:00:03 2021

@author: Jesus Salazar Araya, Angello Crawford Clark
"""

#Librerias utilizadas
import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt

#Ecuacion Temporal de la ecuacion de difusion
def ec_temporal(t,D,Lx,n):
    temporal = np.exp(-D*t*(n*np.pi/Lx)**2)
    return temporal

#Ecuacion Espacial de la ecuacion de difusion
def ec_espacial(x,Lx,n):
    espacial = np.sin(n*np.pi*x/Lx)
    return espacial

#Coeficientes de la serie de Fourier
def coeficientes(x0,l,n,x,A,Lx):
    funcion = lambda x:((np.exp(-(((x-x0)**2)/l)))*np.sin(n*np.pi*x/Lx))
    integral, err = spint.quad(funcion,0,Lx)
    coef = (2*A/Lx)*integral
    return coef


#Constantes
A = 2.0
l = 1.5
x0 = 5
D = 0.5
Lx = 10
Lt = Lx
cant_puntos = 150

#Valores de tiempo
t = np.linspace(0,Lt,cant_puntos)

#Valores de posicion
x = np.linspace(0,Lx,cant_puntos)

#Se define la cantidad de sumas que se realizaran en la serie de Fourier.
n = 60

#Se define la funcion difusion con el tamaño adecuado de arreglo.
rho_x_t = np.zeros((len(x),len(t)))

#En estos ciclos for se realiza el calculo de la difusion.
#k representa los valores de tiempo
#h representa los valores de posicion
#i representa los valores desde 1 hasta n que toma la sumatoria de Fourier
#Por ende la funcion rho_x_t recorre la malla de puntos en el eje del tiempo y el eje de posicion y calcula una aproximacion de la solucion para cada punto. 
for k in range(0,len(t)):
    for h in range(0,len(x)):
        for i in range(1,n):
            rho_x_t[k,h] =  rho_x_t[k,h] + coeficientes(x0,l,i,x[h],A,Lx)*ec_espacial(x[h],Lx,i)*ec_temporal(t[k],D,Lx,i)
     

#Este es el procedimiento que se utiliza pra graficar la solucion de la aproximacion de la difusion rho_x_t en el tiempo y el espacio.
X, T = np.meshgrid(x,t)
plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('Difusion')
ax.plot_surface(T,X,rho_x_t, rstride=1,cstride=1,cmap='cividis',edgecolor = 'none')
ax.set_title('Aproximacion de la difusion por series de Fourier')
plt.show()