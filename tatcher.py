import numpy as np
import cv2

mouth_cascade = cv2.CascadeClassifier('tatcher-venv/Scripts/haarcascade_mcs_mouth.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread("elon.jpg")
# image = cv2.resize(image, (900, 800), cv2.INTER_NEAREST)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.4, minNeighbors=4)

if len(faces) == 0:
    print("No faces detected.")
else:
    # Iterate through the detected faces and process each face
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Identify eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                eye_flipped = cv2.flip(eye_roi, 0)  # Flip the eye vertically
                roi_color[ey:ey+eh, ex:ex+ew] = eye_flipped

        # Identify mouth within the face ROI
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        if len(mouth) > 0:
            for (mx, my, mw, mh) in mouth:
                #cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 5)
                mouth_roi = roi_color[my:my+mh, mx:mx+mw]
                mouth_flipped = cv2.flip(mouth_roi, 0)  # Flip the mouth vertically
                roi_color[my:my+mh, mx:mx+mw] = mouth_flipped

# cv2.imshow('Face, Eyes, and Mouth Detected and Flipped', image)
cv2.imwrite('elon-problem.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
