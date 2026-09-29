import cv2
import numpy as np
cam = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    sucess, frame= cam.read()
    if not sucess:
        print("Cannot read frame")
        break
    frame= cv2.flip(frame,1)
    height,width=frame.shape[:2]
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(30,30))
    #Create a funny image
    map_x=np.zeros((height,width),dtype=np.float32)
    map_y=np.zeros((height,width),dtype=np.float32)
    for y in range(height):
        for x in range(width):
            #Normalize the axes
            nx=(x-width/2)/(width/2)
            ny=(y-height/2)/(height/2)
            #3d fish eye effect
            r=np.sqrt(nx*nx+ny*ny)
            factor=1+0.25*r*r
            new_x=nx*factor
            new_y=ny*factor
            map_x[y,x]=((new_x+1)*width/2)
            map_y[y,x]=((new_y+1)*height/2)
    #Apply the distortion
    funny=cv2.remap(frame,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    #     cv2.putText(frame,f"Faces: {len(faces)}",(20,45),cv2.FONT_ITALIC,1,(255,90,0),3)
    cv2.imshow("Face Detection",funny)
    cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()
