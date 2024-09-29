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
                ex_crop = max(ex + 10, 0)
                ey_crop = max(ey + 10, 0)
                ew_crop = min(ew - 20, roi_color.shape[1] + ex_crop)
                eh_crop = min(eh - 20, roi_color.shape[0] + ey_crop)

                eye_roi = roi_color[ey_crop:ey_crop+eh_crop, ex_crop:ex_crop+ew_crop]
                eye_flipped = cv2.flip(eye_roi, 0)
                roi_color[ey_crop:ey_crop+eh_crop, ex_crop:ex_crop+ew_crop] = eye_flipped

        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=8)
        if len(mouth) > 0:
            for (mx, my, mw, mh) in mouth:
                mx_crop = max(mx + 10, 0)
                my_crop = max(my + 10, 0)
                mw_crop = min(mw - 20, roi_color.shape[1] + mx_crop)
                mh_crop = min(mh - 20, roi_color.shape[0] + my_crop)

                mouth_roi = roi_color[my_crop:my_crop+mh_crop, mx_crop:mx_crop+mw_crop]
                mouth_flipped = cv2.flip(mouth_roi, 0)
                roi_color[my_crop:my_crop+mh_crop, mx_crop:mx_crop+mw_crop] = mouth_flipped
image = cv2.flip(image, 0)

cv2.imwrite('elon-after.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
