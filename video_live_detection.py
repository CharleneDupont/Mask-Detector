import cv2
import sys
import os

import tensorflow as tf
import numpy
from keras import models

#Chemin d'accès au modèle de détection du visage
faceCascade = cv2.CascadeClassifier('static/models/haarcascade_frontalface_default.xml')

#ouverture de la caméra
video_capture = cv2.VideoCapture(0)

#Chargement de notre model entrainé pour la détection de masque
model = models.load_model('static/models/model2.h5')

i=0

#Tant que la caméra est active
while True:
    # Capture de chaque image de la caméra en live
    ret, frame = video_capture.read()
    
    #Passage de l'image en niveau de gris pour que le modèle de détection du isage puisse l'interpréter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detection des visages de l'image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    # Pour chaque visage détecté, prévision de si il y a un masque où non
    for (x, y, w, h) in faces:
        i = i+1
        #Pour chaque visage détecter on enregitrer la partie de l'image détectée
        crop_img = frame[y:int(round(y+h*1.2)), x:int(round(x+w*1.2))]
        face = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (192, 192))
        img_tf = tf.constant(face)
        img_rezize = img_tf/255
        img_rezize = tf.expand_dims(img_rezize, axis=0)
        prediction = model.predict_classes(img_rezize.numpy())
        #fullname = 'static/test/myFace'+str(i)+'_'+str(prediction[0])+'.jpg'
        #cv2.imwrite(fullname, crop_img)
        #print(prediction)
        pred_number = model.predict(img_rezize.numpy())

        
        if prediction[0]==0:
            pred_mask = "No mask"
            cv2.rectangle(frame, (x, y), (int(round(x+w)), int(round(y+h))), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (int(round(x+w)), int(round(y-20))), (0, 0, 255), -1)
            cv2.putText(frame,pred_mask+' : '+str(numpy.round(1-pred_number.flatten()[0],3)), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)
            #cv2.putText(frame,pred_mask, (x,y), 1, 1.4, (25, 24, 55))
            #cv2.putText(frame,pred_number, (x,(y+10)), 1, 1.4, (25, 24, 55))
        elif prediction[0]==1:
            pred_mask = "Mask"
            cv2.rectangle(frame, (x, y), (int(round(x+w)), int(round(y+h))), (84, 222, 114), 2)
            cv2.rectangle(frame, (x, y), (int(round(x+w)), int(round(y-20))), (84, 222, 114), -1)
            cv2.putText(frame,pred_mask+' : '+str(numpy.round(pred_number.flatten()[0],3)), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)
            #cv2.putText(frame,pred_number, (x,(y+10)), 1, 1.4, (25, 24, 55))
        else :
            print("ERREUR")   
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

print(x)
print(y)
print(pred_number)
'''
img_bytes =tf.io.read_file('static/upload/nomask1.jpg')
img_tf = tf.image.decode_jpeg(img_bytes)
img_rezize = tf.image.resize(img_tf, [192,192])
img_rezize = tf.expand_dims(img_rezize, axis=0)
# Scaling
#data = data.astype(‘float’) / 255
#print("Prédiction du model : ", model.predict_classes(img_rezize.numpy()))    

# Prediction
result = model.predict_classes(img_rezize.numpy())
print(result)
'''