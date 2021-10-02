import cv2
import os

dirActual=os.getcwd()
dirModelo=dirActual+"\haarcascade_frontalface_default.xml"
dirImagen=dirActual+"\personas.jpg"

# Importo el modelo entrenado
clasificar = cv2.CascadeClassifier(dirModelo)
# Importo la imagen
imagenOriginal=cv2.imread(dirImagen)
# Convierto a Gris
imageGris=cv2.cvtColor(imagenOriginal,cv2.COLOR_BGR2GRAY)

# Clasificar
caras = clasificar.detectMultiScale(imageGris,
  scaleFactor=1.1,
  minNeighbors=1,
  minSize=(30,30),
  maxSize=(400,400))
for (x,y,w,h) in caras:
  cv2.rectangle(imagenOriginal,(x,y),(x+w,y+h),(0,255,0),2)

# Mostrar
cv2.imshow('Detector',imagenOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()