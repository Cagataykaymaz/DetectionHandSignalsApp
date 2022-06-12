from contextlib import closing
import cv2
from cv2 import RETR_TREE
from cv2 import CHAIN_APPROX_NONE
import numpy as np
import os

camera =cv2.VideoCapture(0,cv2.CAP_DSHOW)
kernel =np.ones((12,12),np.uint8)



name="merhaba_1"

while True:
    ret,pixel= camera.read()
    
    cut_Pixel = pixel[0:200,0:250]
    cut_Pixel_GRAY =cv2.cvtColor(cut_Pixel,cv2.COLOR_BGR2GRAY)
    cut_Pixel_HSV = cv2.cvtColor(cut_Pixel,cv2.COLOR_BGR2HSV)  
    
    lower_Values = np.array([0,20,40])
    upper_Values= np.array([40,255,255])
      
    color_Filter =cv2.inRange(cut_Pixel_HSV,lower_Values,upper_Values)
    color_Filter =cv2.morphologyEx(color_Filter,cv2.MORPH_CLOSE,kernel)
    color_Filter =cv2.dilate(color_Filter,kernel,iterations=1)
    result=cut_Pixel.copy()
    
 
    
    cnts , _= cv2.findContours(color_Filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    max_Width =0
    max_Length=0
    max_Index= -1
    
    for t in range(len(cnts)):
        cnt=cnts[t]
        x,y,w,h =cv2.boundingRect(cnt)
        if(w>max_Width and h>max_Length):
            max_Length=h
            max_Width=w
            max_Index=t
    
    if(len(cnts)>0):
        x,y,w,h =cv2.boundingRect(cnts[max_Index])
        cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)  
        hand_Pic = color_Filter[y:y+h,x:x+w]
        cv2.imshow("El Resim",hand_Pic)
   
    cv2.imshow("Kamera",pixel)
    cv2.imshow("Kesilmiş kare",cut_Pixel)
    cv2.imshow("Renk filtresi",color_Filter)
   
    cv2.imshow("Sonuc",result)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.imwrite("dataset/"+name+".jpg",hand_Pic)    
    
camera.release()  
cv2.destroyAllWindows()  
