import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math



Img1 = cv2.imread('Imagen_Dia.jpg')
Img2 = cv2.imread('Imagen_Noche.jpg')
Img3 = cv2.imread('Imagen_Noche.jpg',0)


res1 = cv2.resize(Img1, dsize=(380, 380))
cv2.imshow('Img1',res1)

res2 = cv2.resize(Img2, dsize=(380, 380))
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

    #Raiz
    #Raiz = math.sqrt(res1)
    #cv2.imshow('Raiz',Raiz)
    #cv2.waitKey()
    #cv2.destroyWindow('Raiz')




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



    
    
cv2.waitKey()
cv2.destroyAllWindows()



