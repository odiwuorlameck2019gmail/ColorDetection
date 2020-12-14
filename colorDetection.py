import cv2 
import numpy as np 

#Capture the video through the webcam.
cap=cv2.VideoCapture(0)
#Colour Detection.
cv2.namedWindow("Colour-Detection",cv2.WINDOW_NORMAL)
while True:
    success,imageFrame=cap.read()
    imageFrame=cv2.flip(imageFrame,1)
    #Convert the image frame to hsv.
    hsvFrame=cv2.cvtColor(imageFrame,cv2.COLOR_BGR2HSV)

    #Set the range for red color and define  the mask.
    red_lower=np.array([136,87,111],np.uint8)
    red_upper=np.array([180,255,255],np.uint8)
    red_mask=cv2.inRange(hsvFrame,red_lower,red_upper)

    #Set range for green color  and se the mask .
    green_lower=np.array([25,52,72],np.uint8)
    green_upper=np.array([102,255,255],np.uint8)
    green_mask=cv2.inRange(hsvFrame,green_lower,green_upper)

    #Set the range of blue color and define the mask .
    blue_lower=np.array([94,80,2],np.uint8)
    blue_upper=np.array([120,255,255],np.uint8)
    blue_mask=cv2.inRange(hsvFrame,blue_lower,blue_upper)

    #create the kernel to be used for the dilation.
    kernel=np.ones((5,5),np.uint8)

    #Dilate red color .
    red_mask=cv2.dilate(red_mask,kernel)
    res_red=cv2.bitwise_and(imageFrame,imageFrame,mask=red_mask)

    #Dilate green color.
    green_mask=cv2.dilate(green_mask,kernel)
    res_green=cv2.bitwise_and(imageFrame,imageFrame,mask=green_mask)

    #Dilate blue color.
    blue_mask=cv2.dilate(blue_mask,kernel)
    res_blue=cv2.bitwise_and(imageFrame,imageFrame,mask=blue_mask)

    #Create contour to track red color 
    contours,hierachy=cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area=cv2.contourArea(contour)
        if area>300:
            x,y,w,h=cv2.boundingRect(contour)
            imageFrame=cv2.rectangle(imageFrame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(imageFrame,"Red Colour",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),1)

    #Create contour to track green color .
    contours,hierachy=cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area=cv2.contourArea(contour)
        if area>300:
            x,y,w,h=cv2.boundingRect(contour)
            imageFrame=cv2.rectangle(imageFrame,(x,y),(w+x,h+y),(0,255,0),2)
            cv2.putText(imageFrame,"Green Colour",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),1)
    #Creating contour to track blue color .
    contours,hierachy=cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area=cv2.contourArea(contour)
        if(area>300):
            x,y,w,h=cv2.boundingRect(contour)
            imageFrame=cv2.rectangle(imageFrame,(x,y),(w+x,h+y),(255,0,0),1)
            cv2.putText(imageFrame,"Blue Colour",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),1)
    imageStack=np.hstack((res_blue,res_green))
    imageStack2=np.hstack((res_red,res_blue))
    allstack=np.vstack((imageStack,imageStack2))
    cv2.imshow("Regeons-of-Colors-Deteted with zero noise:",cv2.resize(allstack,(600,500)))
    cv2.imshow("Colour-Detection",imageFrame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break






