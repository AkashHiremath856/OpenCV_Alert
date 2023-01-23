import cv2 as cv
from playsound import playsound
import time


def play_():
    playsound('yo.wav', False)
    time.sleep(5)


capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    haar_cascade_face = cv.CascadeClassifier(
        'haarcascades\haarcascade_frontalface_default.xml')

    haar_cascade_eye = cv.CascadeClassifier(
        'haarcascades\haarcascade_eye.xml')

    face_rect = haar_cascade_face.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=1)

    eye_rect = haar_cascade_eye.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=1)

    for (x, y, w, h) in face_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    for (x, y, w, h) in eye_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

    if len(face_rect) > 0:
        print(f'Total faces are {len(face_rect)}')
        print(f'eyes detected {len(eye_rect)}')
        if len(eye_rect) < 1:
            play_()
    cv.imshow('video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)
