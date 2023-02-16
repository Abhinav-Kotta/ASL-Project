import cv2
import mediapipe as mp
from ASL_module import HandDetector

cap = cv2.VideoCapture(0)
handDetector = HandDetector()

while True:
    success, img = cap.read()

    if not success:
        break

    image = cv2.flip(img, 1)

    image = handDetector.detect_hands(image)
    landmark_list = handDetector.detect_position(image)

    if handDetector.letter_A_test():
        print('A')
    elif handDetector.letter_B_test():
        print('B')
    elif handDetector.letter_C_test():
        print('C')

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

cap.release()
cv2.destroyAllWindows()
