import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, hands_to_track=2, detection_confidence=0.5, tracking_confidence=0.5) -> None:
        self.mode = mode
        self.hands_to_track = hands_to_track
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands(static_image_mode=self.mode, max_num_hands=self.hands_to_track,
                                        min_detection_confidence=self.detection_confidence, min_tracking_confidence=self.tracking_confidence)

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.fingertip_ids = [4, 8, 12, 16, 20]

    def detect_hands(self, image, draw=True):
        self.result = self.hands.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks,
                                            self.mp_hand.HAND_CONNECTIONS,
                                            self.mp_styles.get_default_hand_landmarks_style(),
                                            self.mp_styles.get_default_hand_connections_style())

        return image

    def detect_position(self, image, hand_index=0, draw=True):
        self.landmark_list = []

        if self.result.multi_hand_landmarks:
            main_hand = self.result.multi_hand_landmarks[hand_index]

            for id, landmark in enumerate(main_hand.landmark):
                h, w, c = image.shape

                cx, cy = int(landmark.x*w), int(landmark.y*h)
                self.landmark_list.append([id, cx, cy])

                cv2.putText(image, str(id), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)

        return self.landmark_list

    def letter_A_test(self):
        if len(self.landmark_list) != 0:
            for tip_ids in range(1, 5):
                if self.landmark_list[self.fingertip_ids[tip_ids]][2] > self.landmark_list[self.fingertip_ids[tip_ids] - 3][2]:
                    return True

        return False

    def letter_B_test(self):
        if len(self.landmark_list) != 0:
            for tip_ids in range(1, 5):
                if self.landmark_list[self.fingertip_ids[tip_ids]][2] < self.landmark_list[self.fingertip_ids[tip_ids] - 3][2]:
                    if self.landmark_list[self.fingertip_ids[0]][1] > self.landmark_list[self.fingertip_ids[tip_ids]-2][1]:
                        return True

        return False

    def letter_C_test(self):
        if len(self.landmark_list) != 0:
            for tip_ids in range(1, 5):
                if self.landmark_list[self.fingertip_ids[tip_ids]][2] > self.landmark_list[self.fingertip_ids[tip_ids] - 2][2]:
                    if self.landmark_list[self.fingertip_ids[0]][1] < self.landmark_list[self.fingertip_ids[tip_ids]-2][1]:
                        return True

        return False
