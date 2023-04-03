import cv2 as cv
import matplotlib.pyplot as mat

#plot class 
def plot(img):
    mat.imshow(img, cmap="gray")
    mat.axis('off')
    mat.style.use('seaborn')
    mat.show()

#read and convert the image from BGR to RGB
path=input("enter image path : \n")
img=cv.imread(path)
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

#face detection
face=cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
data=face.detectMultiScale(img, 1.05, 10)

#face bluration
for (x,y,w,h) in data:
    cv.rectangle(img,(x,y),(x+w,y+h),(230,230,250),5)
    ROI=img[y:y+h,x:x+w]
    ROI=cv.GaussianBlur(ROI,(25,25),35)
    img[y:y+ROI.shape[0],x:x+ROI.shape[1]]=ROI

plot(img)

