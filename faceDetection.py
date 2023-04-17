import cv2

faceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # read trained data

webcam = cv2.VideoCapture(0) # choose webcam device

while True:
   frameRead, frame = webcam.read() # get webcam frames
   grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frames black and white

   faceCords = faceData.detectMultiScale(grayFrame) # detect faces

   for (x, y, w, h) in faceCords:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4) # draw rectangle over face
      cv2.waitKey(1)

   cv2.imshow("Webcam face detector", frame) # show result
   key = cv2.waitKey(1)

   if key == 113:
      break
