import cv2
import mediapipe as mp
import pyautogui as pg

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

pg.FAILSAFE = False

command_status = False
pinch_coordinates = None

def distance(dot1, dot2, landmarks):
    x_dot1 = landmarks.landmark[dot1].x
    y_dot1 = landmarks.landmark[dot1].y

    x_dot2 = landmarks.landmark[dot2].x
    y_dot2 = landmarks.landmark[dot2].y

    return_distance = ((x_dot1 - x_dot2) ** 2 + (y_dot1 - y_dot2) ** 2) ** 0.5
    return return_distance

while cap is not None:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Draw landmarks
                if idx == 8 or idx == 4 or idx == 12:
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)  # Green dots for thumb, index and middle
                else:
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)  # Red dots for other finger landmarks

                if idx == 8 and command_status:
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)  # Yellow dots for pinched index

            pinch_threshold = 0.04  # You can adjust this threshold for own purpose(pinch recongnization length)
            if landmarks.landmark[0].x < 0.5:
                if distance(4, 8, landmarks) < pinch_threshold:
                    if (
                        distance(4, 11, landmarks) > pinch_threshold
                        and distance(4, 15, landmarks) > pinch_threshold
                        and distance(4, 12, landmarks) > pinch_threshold
                    ):
                        # Pinch gesture recognized by the left hand only
                        if not command_status:
                            command_status = True
                            pinch_x = landmarks.landmark[0].x
                            pinch_y = landmarks.landmark[0].y
                            print("Pinch gesture detected", pinch_x, pinch_y)
                else:
                    command_status = False
                    pinch_coordinates = None  # Reset pinch_coordinates when pinch is released
            
            if pinch_coordinates is None and command_status:
                pinch_x, pinch_y = int(landmarks.landmark[8].x * w), int(landmarks.landmark[8].y * h)
                pinch_coordinates = (pinch_x, pinch_y)
            elif pinch_coordinates is not None and command_status:
                pg.moveRel(0.5*(landmarks.landmark[8].x*w - pinch_x), 0.5*(landmarks.landmark[8].y*h - pinch_y), duration=0.1)
                


    cv2.imshow('Hand Tracking', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
