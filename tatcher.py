import numpy as np
import cv2

mouth_cascade = cv2.CascadeClassifier('tatcher-venv/Scripts/haarcascade_mcs_mouth.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread("elon.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.6, minNeighbors=8)

if len(faces) == 0:
    print("No faces detected.")
else:
    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                eye_flipped = cv2.flip(eye_roi, 0)
                roi_color[ey:ey+eh, ex:ex+ew] = eye_flipped

        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=8)
        if len(mouth) > 0:
            for (mx, my, mw, mh) in mouth:
                mouth_roi = roi_color[my:my+mh, mx:mx+mw]
                mouth_flipped = cv2.flip(mouth_roi, 0)
                roi_color[my:my+mh, mx:mx+mw] = mouth_flipped

cv2.imwrite('elon-after.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
