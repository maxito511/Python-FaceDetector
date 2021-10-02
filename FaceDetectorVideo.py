import cv2
import numpy as np
import os

dirActual=os.getcwd()
dirModelo=dirActual+"\haarcascade_frontalface_default.xml"
dirImagen=dirActual+"\personas.jpg"
# Capturo video
video = cv2.VideoCapture(0)
# Importo el modelo entrenado
clasificar = cv2.CascadeClassifier(dirModelo)

while True:
    _,captura = video.read()
    videoGris=cv2.cvtColor(captura,cv2.COLOR_BGR2GRAY)

    # Mostrar
    caras = clasificar.detectMultiScale(videoGris,
    scaleFactor=1.1,
    minNeighbors=5)
    for (x,y,w,h) in caras:
        cv2.rectangle(captura,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('Detector',captura)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Mostrar
video.release()
cv2.destroyAllWindows()