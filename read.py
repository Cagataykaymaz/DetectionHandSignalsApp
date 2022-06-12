import cv2
import numpy as np
import os

camera =cv2.VideoCapture(0,cv2.CAP_DSHOW)
kernel =np.ones((12,12),np.uint8)



def pictureFindDiff(pic1,pic2):
    pic2 =cv2.resize(pic2,(pic1.shape[1],pic1.shape[0]))
    pictureDiff = cv2.absdiff(pic1,pic2)
    numberDiff=cv2.countNonZero(pictureDiff)
    return numberDiff
  

def uploadData():
    data_Names = []
    data_Pics =[]
    
    files = os.listdir("dataset/")
    
    for file in files:       
        data_Names.append(file.replace(".jpg",""))
        data_Pics.append(cv2.imread("dataset/"+file,0))

        
    return data_Names,data_Pics

def classify(picture,data_Names,data_Pics):
    min_Index = 0
    min_Value = pictureFindDiff(picture,data_Pics[0])
    for t in range(len(data_Names)):
        diff_Value = pictureFindDiff(picture,data_Pics[t])
        if(diff_Value<min_Value):
            min_Value=diff_Value
            min_Index=t
    return data_Names[min_Index]

data_Names,data_Pics = uploadData()        

while True:
    ret,pixel= camera.read()
    cut_Pixel = pixel[0:200,0:250]
    frame=pixel[0:500,0:500]
    cut_Pixel_Gray =cv2.cvtColor(cut_Pixel,cv2.COLOR_BGR2GRAY)
    cut_Pixel_HSV = cv2.cvtColor(cut_Pixel,cv2.COLOR_BGR2HSV)  
    
    lower_Values = np.array([0,40,40])
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
       # cv2.imshow("El Resim",El_Resim)
        word=classify(hand_Pic,data_Names,data_Pics)
        new_word=''
        i=0
        for x in word:
            if(word[i]!='_'):
                new_word+=word[i]
                i=i+1
            else:
                i=0
                break    
                
                
      
        cv2.putText(frame,new_word.upper(),(250,80),cv2.FONT_HERSHEY_DUPLEX,2,(150,44,76),2)
        
       
    cv2.imshow("Kamera",pixel)
   
    cv2.imshow("Renk filtresi",color_Filter)
   
    cv2.imshow("Sonuc",result)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
camera.release()  
cv2.destroyAllWindows()  
