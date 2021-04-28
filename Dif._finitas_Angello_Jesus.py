# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:41:16 2021

@author: Jesus Salazar Araya, Angello Crawford Clark
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 




def AproxDIFXY(difxy, pm_x,pm_y, prec, D, delta_x,delta_y):
    '''Calcula el valor aproximado de la difusion en el punto (x, y)
 
    Parámetros de la función
    ------------------------
    difxy : matriz con los valores iniciales de la difusion en cada
           punto de la malla
    pm_x : número de puntos de la malla en el eje x
    pm_y : número de puntos de la malla en el eje y
    prec : precisión requerida para el cálculo aproximado de los valores de la difusion en la malla
    D : Constante que se nos brinda en el enunciado
    delta_x : espaciamiento delta en el eje x
    delta_y : espaciamiento delta en el eje y
 
    Salida de la función
    --------------------
    valorAproxUXY : matriz con los valores finales de la difusion
                    en cada punto de la malla
    '''

    # Se define el contador de iteraciones
    contador_iteraciones = 0

    # Se define una variable de control como 1 para que se pueda ejecutar el metodo iterativo 
    # Esta variable controla la continuidad del ciclo 
    canasta_imprec = 1

    # Se realiza el cálculo iterativo de la difusion aproximado en cada punto
    # de la malla hasta que se alcanza la precisión requerida
    
    while canasta_imprec > 0:
      # Se aumenta el contador de interaciones en 1 unidad por cada vez que se realiza el ciclo
      contador_iteraciones += 1

      # Se define la variable de control como 0 en caso de que no se necesite ejecutar el ciclo otra vez
      canasta_imprec = 0

      for n in range(0, pm_y-1, 1):
        for m in range(1, pm_x-1, 1):
          # Se crea una matriz a partir de la matriz ingresada 
          difxy_anterior = difxy[m, n]
          
          # Se aplica el método de Gauss-Siedel para esta EDP particular
          difxy[m, n+1] =  difxy_anterior + (D*delta_y*(difxy[m+1,n]+difxy[m-1,n]-2*difxy[m,n]))/(delta_x**2)
          

          # Se calcula la diferencia finita de las matriz resultante de Gauss-Seidel menos la matriz anterior
          dif_finita = np.abs([difxy[m,n+1]-difxy_anterior])[0]
          
          # Se compara el valor de la diferencia con la precisión y si esta diferencia no logra
          # estar dentro de la precisión indicada se vuelve a ejecutar el ciclo
          if dif_finita > prec:
            canasta_imprec += 1

          # Se define una condicion para que se detenga el método iterativo al máximo de 500 iteraciones
          if contador_iteraciones > 500:
            canasta_imprec = 0
     
          
    #Se imprime la cantidad de interaciones alcanzadas al llevar a cabo el método
    valorAproxDIFXY = difxy
    print("Cantidad iteraciones para alcanzar precisión: ", contador_iteraciones)
    return valorAproxDIFXY

Lx = 10.0
Lt = 10.0

def Grafico_interactivo_difusion(Lx, prec):
  '''Calcula el valor aproximado de la difusion en una malla de puntos
  y lo representa en un gráfico
 
  Parámetros de la función
  ------------------------
  lx : arista de la placa cuadrada sujeta al potencial eléctrico
  prec : precisión requerida para el cálculo aproximado de los valores del
            potencial en la malla
 
  Salida de la función
  --------------------
  Z : conjunto de valores de la difusion aproximado en los puntos de la malla
      y gráfico 3D
  '''
 
  # Se define la malla de puntos para evaluar en la difusion y se anotan otros parametros que se utilizaran
  A=2.0
  l=1.5
  X0=5.0
  D=0.5
  puntosmalla_x = 50
  puntosmalla_y = 500
  delta_x = Lx/puntosmalla_x
  delta_t = Lt/puntosmalla_y
  x = np.linspace(0, Lx, puntosmalla_x)
  y = np.linspace(0, Lt, puntosmalla_y)
  
  
  # Se inicializa la matriz con solo ceros que va a representar la difusion
  
  difxyi = np.zeros((puntosmalla_x, puntosmalla_y), float)

  # Se establecen las condiciones de frontera 
  
  for i in range(0, puntosmalla_x):
    difxyi[i,0] = A*np.exp((-(x[i]-X0)**2)/l)

  # Se calcula el valor aproximado de la difusion en los puntos de
  # la malla y se asignan al eje Z
  Z = AproxDIFXY(difxyi, puntosmalla_x, puntosmalla_y, prec, D, delta_x,delta_t)    
  
  
  
  X,T = np.meshgrid(x, y)
  plt.figure(figsize=(10,6))
  ax = plt.axes(projection='3d')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_zlabel('Difusión')
  ax.plot_surface(T,X, Z.T, rstride=1, cstride=1, cmap='cividis', edgecolor='none')
  ax.set_title('Aproximacion de la difusion por diferencias finitas')
  plt.show()
  return

Q = Grafico_interactivo_difusion(Lx, 0.001)




