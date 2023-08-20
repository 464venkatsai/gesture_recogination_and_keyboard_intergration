import mediapipe as mp
import cv2
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
def distance(point1,point2) : return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

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

def check(list_name,symbol) : 
     return all([True if dist<=list_name[i] else False for i,dist in enumerate(distances)]) if symbol == '<=' else all([True if dist>=list_name[i] else False for i,dist in enumerate(distances)])
    
fist_close = [0.38,0.25,0.2,0.19,0.2]
fist_open = [0.24,0.39,0.43,0.41,0.35]
while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            fingers = fingers_landmarks(hand)
            distances = [distance(fingers[20],fingers[i]) for i in range(0,20,4)]
            if check(fist_close,'<='):
                 keyboard.press('left')
            elif check(fist_open,'>='):
                 keyboard.press('right')
            else:
                 keyboard.release('right')
                 keyboard.release('left')
            
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'):
        print(image.shape)
        break

cap.release()
cv2.destroyAllWindows()