import mediapipe as mp
import cv2
import keyboard
import time

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

def distance(point1,point2):
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

def Find_hand(point1,point2,image):
    x_pixel, y_pixel = int(point1.x * image.shape[1]), int(point1.y * image.shape[0]) 
    if point1.x > point2.x :
        cv2.putText(image, "Right Hand", (x_pixel,y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else :
        cv2.putText(image, "Left Hand", (x_pixel,y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def fingers_landmarks(hand):
        thumbpoint1  = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumbpoint2  = hand.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumbpoint3  = hand.landmark[mp_hands.HandLandmark.THUMB_MCP]
        thumbpoint4  = hand.landmark[mp_hands.HandLandmark.THUMB_CMC]
        indexpoint1  = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        indexpoint2  = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
        indexpoint3  = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        indexpoint4  = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middlepoint1 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middlepoint2 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        middlepoint3 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middlepoint4 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ringpoint1   = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ringpoint2   = hand.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
        ringpoint3   = hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        ringpoint4   = hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        littlepoint1 = hand.landmark[mp_hands.HandLandmark.PINKY_TIP]
        littlepoint2 = hand.landmark[mp_hands.HandLandmark.PINKY_DIP]
        littlepoint3 = hand.landmark[mp_hands.HandLandmark.PINKY_PIP]
        littlepoint4 = hand.landmark[mp_hands.HandLandmark.PINKY_MCP]
        wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
        return [thumbpoint1,thumbpoint2,thumbpoint3,thumbpoint4,indexpoint1,indexpoint2,indexpoint3,indexpoint4,middlepoint1,middlepoint2,middlepoint3,middlepoint4,ringpoint1,ringpoint2,ringpoint3,ringpoint4,littlepoint1,littlepoint2,littlepoint3,littlepoint4,wrist]


while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for num , hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                # if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                #     left_hand_landmarks = fingers_landmarks(results.multi_hand_landmarks[0])
                #     right_hand_landmarks = fingers_landmarks(results.multi_hand_landmarks[1])
                #     distances_left = [distance(left_hand_landmarks[20], point) for point in left_hand_landmarks]
                #     distances_right = [distance(right_hand_landmarks[20], point) for point in right_hand_landmarks]
                #     print([distances_left[i] for i in range(0,20,4)])
                #     print([distances_right[i] for i in range(0,20,4)])
                fist_close = [0.38,0.25,0.2,0.19,0.2]
                fist_open = [0.24,0.39,0.43,0.41,0.35]
                fingers = fingers_landmarks(hand)
                if [distance(fingers[20],fingers[i]) for i in range(0,20,4)] <=fist_close:
                    print('left')
                
                if [distance(fingers[20],fingers[1]) for i in range(0,20,4)]>= fist_open:
                    print('right')


                # Fist Closed
                # if distances_point1[0]<=0.38 and distances_point1[1]<=0.25 and distances_point1[2]<=0.2 and distances_point1[3]<=0.19 and distances_point1[4]<=0.2:
                #     keyboard.press('alt+tab')

                # # Fist Opened
                # elif distances_point1[0]>=0.24 and distances_point1[1]>=0.39 and distances_point1[2]>=0.43 and distances_point1[3]>=0.41 and distances_point1[4]>=0.35:
                #     keyboard.press_and_release('enter')
                #     print('right')

                # else:
                #     keyboard.release('alt+tab')
                #     # keyboard.release('left')
        
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'):
        print(image.shape)
        break

cap.release()
cv2.destroyAllWindows()
