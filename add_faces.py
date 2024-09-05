import cv2
import os
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input("Enter Your Name: ")
os.makedirs(f'data/faces/{name}', exist_ok=True)

i = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))
        cv2.imwrite(f'data/faces/{name}/{i}.jpg', resized_img)
        i += 1
        cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or i == 100:
        break

video.release()
cv2.destroyAllWindows()
print(f"A new user with the name '{name}' has been added.")