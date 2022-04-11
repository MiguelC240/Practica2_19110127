import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import cv2
import math

import numpy as np
import skimage
from skimage import io


Img1 = cv2.imread('Imagen_Dia.jpg')
Img2 = cv2.imread('Imagen_Noche.jpg')
Img3 = cv2.imread('Imagen_Noche.jpg',0)


res1 = cv2.resize(Img1, dsize=(280, 280))
cv2.imshow('Img1',res1)

res2 = cv2.resize(Img2, dsize=(280, 280))
cv2.imshow('Img2',res2)


Img_Negro = cv2.resize(Img3, dsize=(380, 380))




M = cv2.waitKey (0) & 0xFF
if M == ord('m'):
###################### SUMA ##############################
    
    #Suma
    Suma = res1 + res2
    cv2.imshow('Suma',Suma)
    cv2.waitKey()
    cv2.destroyWindow('Suma')

    #Adicion
    Adicion = cv2.add(res1,res2)
    cv2.imshow('Adicion',Adicion)
    cv2.waitKey()
    cv2.destroyWindow('Adicion')

    #Adicion
    Adi = cv2.addWeighted(res1,0.5,res2,0.5,0)
    cv2.imshow('Adi',Adi)
    cv2.waitKey()
    cv2.destroyWindow('Adi')

###################### RESTA ##############################

    #Resta
    Resta = res1 - res2
    cv2.imshow('Resta',Resta)
    cv2.waitKey()
    cv2.destroyWindow('Resta')

    #Sustraccion
    Sustraccion = cv2.subtract(res1,res2)
    cv2.imshow('Sustraccion',Sustraccion)
    cv2.waitKey()
    cv2.destroyWindow('Sustraccion')


    #Absdiff
    Absdiff = cv2.absdiff(res1,res2)
    cv2.imshow('Absdiff',Absdiff)
    cv2.waitKey()
    cv2.destroyWindow('Absdiff')


###################### MULTIPLICACION ##############################

    #Multiplicacion
    Multiplicacion = res1 * res2
    cv2.imshow('Multiplicacion',Multiplicacion)
    cv2.waitKey()
    cv2.destroyWindow('Multiplicacion')


    #Multiply
    Multiply = cv2.multiply(res1,res2)
    cv2.imshow('Multiply',Multiply)
    cv2.waitKey()
    cv2.destroyWindow('Multiply')


###################### DIVISION ##############################

    #Division
    Division = res1 / res2
    cv2.imshow('Division',Division)
    cv2.waitKey()
    cv2.destroyWindow('Division')


    #Divide
    Divide = cv2.divide(res1,res2)
    cv2.imshow('Divide',Divide)
    cv2.waitKey()
    cv2.destroyWindow('Divide')
    

###################### LOGARITMO NATURAL ##############################

    #Logaritmo
    Logaritmo = np.zeros(res1.shape, res1.dtype)
    c = 1
    Logaritmo = c * np.log(1+res1)
    maxi = np.amax(Logaritmo)
    Logaritmo = np.uint8(Logaritmo / maxi *255)
    
    cv2.imshow('Logaritmo',Logaritmo)
    cv2.waitKey()
    cv2.destroyWindow('Logaritmo')



###################### RAIZ ##############################

    #Raiz cuadrada
    Raiz = (res1**(0.5))
    cv2.imshow('Raiz',Raiz)
    cv2.waitKey()
    cv2.destroyWindow('Raiz')




###################### DERIVADA ##############################

    #Derivada

    Derivada = cv2.Laplacian(Img_Negro,cv2.CV_64F)
    
    cv2.imshow('Derivada',Derivada)
    cv2.waitKey()
    cv2.destroyWindow('Derivada')

    


###################### POTENCIA ##############################

    #Potencia
    Potencia = np.zeros(res1.shape, res1.dtype)
    g = 0.5
    Potencia = c * np.power(res1,g)
    maxi1 = np.amax(Potencia)
    Potencia = np.uint8(Potencia/maxi1 * 255)
    
    cv2.imshow('Potencia',Potencia)
    cv2.waitKey()
    cv2.destroyWindow('Potencia')

    #Pow
    Pow = cv2.pow(res1,2)
    cv2.imshow('Pow',Pow)
    cv2.waitKey()
    cv2.destroyWindow('Pow')

    #Potencia
    Pote = (res1**2)
    cv2.imshow('Pote',Pote)
    cv2.waitKey()
    cv2.destroyWindow('Pote')


###################### CONJUNCIÓN ##############################

    #Conjuncion
    conjuncion = cv2.bitwise_and(res1,res2)
    cv2.imshow('Conjuncion', conjuncion)
    cv2.waitKey()
    cv2.destroyWindow('Conjuncion')


###################### DISYUNCIÓN ##############################

    #Disyuncion
    disyuncion = cv2.bitwise_or(res1,res2)
    cv2.imshow('Disyuncion', disyuncion)
    cv2.waitKey()
    cv2.destroyWindow('Disyuncion')


###################### NEGACIÓN ##############################

    #Negación 1
    Negacion = cv2.resize(Img1, dsize=(280, 280))
    height, width, _ = Negacion.shape

    for i in range(0, height - 1):
        for j in range(0, width -1):
            pixel = Negacion[i,j]
            pixel[0] = 255 - pixel[0]
            pixel[1] = 255 - pixel[1]
            pixel[2] = 255 - pixel[2]
            Negacion[i,j] = pixel
    cv2.imshow('Negacion1',Negacion)
    cv2.waitKey()
    cv2.destroyWindow('Negacion1')


    #Negación 2
    Nega = 1 - res1
    cv2.imshow('Negacion2',Nega)
    cv2.waitKey()
    cv2.destroyWindow('Negacion2')


###################### TRASLACIÓN ##############################
    
    #Traslación
    ancho = res1.shape[1] #columnas
    alto = res1.shape[0] # filas
    
    M = np.float32([[1,0,10],[0,1,100]]) #Construccion de la matriz
    traslacion = cv2.warpAffine(res1,M,(ancho,alto))
    cv2.imshow('Traslacion',traslacion)
    cv2.waitKey()
    cv2.destroyWindow('Traslacion')


###################### RESCALADO ##############################

    #Rescalado
    Rescalado = cv2.resize(Img1, dsize=(480, 480))
    cv2.imshow('Rescalado',Rescalado)
    cv2.waitKey()
    cv2.destroyWindow('Rescalado')



###################### ROTACION ##############################

    # Rotación
    Noche = io.imread("Imagen_Noche.jpg")
    type(Noche)
    Noche.shape
    plt.imshow(Noche[::-1])###Invertir imagen
    plt.show()


    # Rotación
    ancho = res2.shape[1] #columnas
    alto = res2.shape[0] # filas
    
    Rotacion = cv2.getRotationMatrix2D((ancho//2,alto//2),180,1)
    imageOut = cv2.warpAffine(res2,Rotacion,(ancho,alto))
    cv2.imshow('Rotacion',imageOut)
    cv2.waitKey()
    cv2.destroyWindow('Rotacion')



###################### TRASLACION A FIN ##############################

    #Traslacion 1
    traslacion1 = cv2.warpAffine(res1,M,(ancho,alto))
    cv2.imshow('Traslacion_1',traslacion1)
    cv2.waitKey()
    cv2.destroyWindow('Traslacion_1')

    #Traslacion 2
    rows, cols, ch = res1.shape
    pts1 = np.float32([[50, 50],
                       [200, 50], 
                       [50, 200]])
      
    pts2 = np.float32([[10, 100],
                       [200, 50], 
                       [100, 250]])

    M2 = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(res1, M2, (cols, rows))
    cv2.imshow('Traslacion_2',dst)
    cv2.waitKey()
    cv2.destroyWindow('Traslacion_2')


    #Traslacion 3
    rows,cols,ch = res1.shape
     
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    M3 = cv2.getPerspectiveTransform(pts1,pts2)
    tras = cv2.warpPerspective(res1,M3,(300,300))
    cv2.imshow('Traslacion_3',tras)
    cv2.waitKey()
    cv2.destroyWindow('Traslacion_3')

###################### TRANSPUESTA ##############################

    #Transpuesta 1
    def transponer(res1):
        t = []
        for i in range(len(res1[0])):
            t.append([])
            for j in range(len(res1)):
                t[i].append(res1[j][i])
        return t
    transpuesta1 = np.concatenate((res1, transponer(res1), res2), axis=1)
    transpuesta1 = cv2.cvtColor(transpuesta1, cv2.COLOR_BGR2RGB)
    cv2.imshow('Transpuesta_1', transpuesta1)
    cv2.waitKey()
    cv2.destroyWindow('Transpuesta_1')

    #Transpuesta 2
    trans2 = cv2.transpose(res1)
    transpuesta2 = np.concatenate((res1, trans2, res2), axis=1)
    transpuesta2 = cv2.cvtColor(transpuesta2, cv2.COLOR_BGR2RGB)
    cv2.imshow('Transpuesta_2', transpuesta2)
    cv2.waitKey()
    cv2.destroyWindow('Transpuesta_2')

    
cv2.waitKey()
cv2.destroyAllWindows()



