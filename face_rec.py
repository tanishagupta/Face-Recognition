import cv2 as cv
cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)


while True:
    ret,frame = cap.read()
    frame = cv.flip(frame,1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces  = cascade.detectMultiScale(gray, 1.3, 5)
    
    for x,y,w,h in faces:
        
        cv.rectangle(frame,(x,y), ((x+w), (y+h)), (255,255,0),2)
        roi_gray  = gray  [y:y+h , x:x+w]
        roi_color = frame [y:y+h , x:x+w]
        
    cv.imshow('Frame',frame)
    
    key = cv.waitKey(30)
    if key == 27:
        break
    
    
cap.release()
cv.destroyAllWindows()
