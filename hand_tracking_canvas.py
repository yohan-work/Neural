import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import mediapipe as mp

# 1. 디바이스 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 2. 모델 정의 (학습 때와 동일)
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EMNIST_CNN(nn.Module):
    def __init__(self):
        super(EMNIST_CNN, self).__init__()
        # 구조는 MNIST와 동일하게 유지하되, Output만 변경
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 26) # 26 Alphabets (0-25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # 모델 로드 (Digit & Alphabet)
    digit_model_path = './mnist_cnn.pth'
    alpha_model_path = './emnist_cnn.pth'
    
    digit_model = MNIST_CNN().to(device)
    alpha_model = EMNIST_CNN().to(device)
    
    models_loaded = {"digit": False, "alpha": False}

    if os.path.exists(digit_model_path):
        try:
            digit_model.load_state_dict(torch.load(digit_model_path, map_location=device))
            digit_model.eval()
            models_loaded["digit"] = True
            print("Digit model loaded.")
        except:
            print("Failed to load digit model.")
    
    if os.path.exists(alpha_model_path):
        try:
            alpha_model.load_state_dict(torch.load(alpha_model_path, map_location=device))
            alpha_model.eval()
            models_loaded["alpha"] = True
            print("Alphabet model loaded.")
        except:
             print("Failed to load alphabet model.")

    if not models_loaded["digit"] and not models_loaded["alpha"]:
        print("Error: No models found. Please train at least one model.")
        return

    # 전처리 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # EMNIST용 전처리 (Rotate & Flip)
    # 학습 때와 달리, 이미 PIL Image로 들어오는 핸드 드로잉 이미지는 정방향이므로
    # 모델이 학습된 방식(90도 회전된 데이터)에 맞춰서 돌려줘서 넣어줘야 함.
    # 학습 데이터: 90도 회전되어 있음 -> 모델이 그걸 학습함.
    # 입력 데이터: 정방향 -> 모델에 넣을 때 90도 회전해서 넣어야 매칭됨.
    emnist_transform_fn = transforms.Compose([
        transforms.functional.hflip,
        lambda x: transforms.functional.rotate(x, -90), # -90 or 90 depends on how data was loaded. Usually EMNIST raw is rotated.
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MediaPipe Hands 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    # 웹캠 설정
    cap = cv2.VideoCapture(0)
    
    # 캔버스 초기화
    canvas = None
    
    # 제스처 쿨다운 및 상태 변수
    last_action_time = 0
    cooldown = 1.0 # 1초 쿨다운
    
    # 팁 ID (엄지, 검지, 중지, 약지, 소지)
    # MediaPipe Hand Landmarks: 
    # 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
    tip_ids = [4, 8, 12, 16, 20]

    # 상태 변수 초기화
    x1, y1 = 0, 0
    prediction_text = ""
    current_mode = "digit" # 'digit' or 'alpha'
    alphabet_map = {i: chr(65+i) for i in range(26)}

    print("Controls (Gestures):")
    print(" - ☝️  Index Up: DRAW")
    print(" - ✌️  Index + Middle Up: HOVER (Move without drawing)")
    print(" - ✊  Fist (All Down): CLEAR")
    print(" - 👍  Thumb Up (Only): PREDICT")
    print(" - 🤘  Rock (Index + Pinky): SWITCH MODE")
    print(" - 'q' to QUIT")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # 거울 모드
        height, width, _ = frame.shape
        
        if canvas is None:
            canvas = np.zeros_like(frame)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        gesture_name = "None"
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 손가락 상태 판별
                fingers = []
                
                # 엄지: x좌표 비교 (오른손 기준, 거울 모드라 반대일 수 있음 확인 필요)
                # 여기서는 간단히 엄지 팁이 엄지 관절(IP, 3)보다 바깥쪽에 있는지보다는 
                # 단순히 y좌표나 x좌표 상대 위치로 해야하는데, 엄지는 회전이 자유로워 까다로움.
                # 편의상 엄지 팁(4)이 엄지 기저부(2)보다 위에 있거나 단순히 펴졌는지 확인.
                # 가장 간단한 방법: 엄지 팁(4)의 x좌표가 새끼손가락 쪽인지 검지 쪽인지 판단.
                # 거울모드(Flip) 상태: 오른쪽 화면이 오른손. 
                # 오른손일 때: 엄지(4)가 관절(3)보다 왼쪽(<)이면 펴진 것. (화면상 왼쪽이 실제 오른쪽)
                # 복잡하므로 엄지는 y좌표가 관절보다 확실히 위에 있는지만 체크하거나, 일단 제외하고 4손가락만 볼 수도 있음.
                # 여기서는 '엄지 팁이 관절(3)보다 x좌표 차이가 크거나' 하는 식으로 많이 하는데,
                # 직관적인 '엄지 척'을 위해 엄지 팁이 검지 관절(5)보다 멀리 떨어져있는지 등으로 판별.
                # 일단 간단한 로직: 엄지 팁이 검지 관절보다 바깥쪽(몸 바깥)에 있음.
                
                # 엄지 (단순화: x좌표가 검지 관절보다 멀리 떨어짐)
                # 오른손/왼손 구분이 없으면 헷갈림.
                # 엄지는 일단 제외하거나, 단순 y좌표로 봅니다 (위로 들었는지).
                # 엄지 팁(4)의 y가 검지 관절(5)의 y보다 작으면 (위에 있으면) Up으로 간주? 
                # 하지만 주먹쥘 때도 그럴 수 있음.
                # 안전하게: 엄지 팁(4)과 새끼 팁(20)의 거리가 멀면 펴진 것?
                
                # 엄지 판별 로직 (x좌표 기반, 오른손 잡이 가정 or hand label check)
                # 여기서는 엄지 제외 4손가락 위주로 하고, 엄지는 별도 제스처로 취급.
                
                # 나머지 4손가락 (검지~소지) : 팁의 y가 관절(Dip)의 y보다 위에(작게) 있으면 펴진 것
                # Landmark: Tip(8,12,16,20), PIP(6,10,14,18) - PIP보다 팁이 위에 있어야 펴진 것
                
                # 검지
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # 중지
                if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
                # 약지
                if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
                # 소지
                if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 엄지 판별 (엄지 척 제스처용): 엄지 팁(4)이 검지 관절(6)보다 상당히 위에 있고, 나머지 손가락은 접힘
                thumb_up = False
                if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y and \
                   hand_landmarks.landmark[4].y < hand_landmarks.landmark[8].y:
                       thumb_up = True

                # 제스처 인식
                # fingers = [검지, 중지, 약지, 소지]
                
                cx, cy = int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)

                # 1. Fist (All Down) -> Clear
                if fingers == [0, 0, 0, 0] and not thumb_up:
                    gesture_name = "Fist (Clear)"
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        canvas = np.zeros_like(frame)
                        prediction_text = ""
                        last_action_time = curr_time
                        print("Canvas Cleared via Gesture")
                    x1, y1 = 0, 0

                # 2. Rock (Index + Pinky Up) -> Switch Mode
                elif fingers == [1, 0, 0, 1]:
                    gesture_name = "Rock (Switch)"
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        current_mode = "alpha" if current_mode == "digit" else "digit"
                        prediction_text = ""
                        last_action_time = curr_time
                        print(f"Switched to {current_mode} via Gesture")
                    x1, y1 = 0, 0

                # 3. Two Fingers (Index + Middle) -> Hover (Move without drawing)
                elif fingers == [1, 1, 0, 0]:
                    gesture_name = "Hover"
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), 2) # 커서 표시
                    x1, y1 = 0, 0 # 선 끊기

                # 4. Only Index Up -> Draw
                elif fingers == [1, 0, 0, 0]:
                    gesture_name = "Draw"
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), cv2.FILLED)
                    if x1 == 0 and y1 == 0:
                        x1, y1 = cx, cy
                    else:
                        cv2.line(canvas, (x1, y1), (cx, cy), (255, 255, 255), 15)
                        x1, y1 = cx, cy

                # 5. Thumb Up (Strict check: others down) -> Predict
                elif thumb_up and fingers == [0, 0, 0, 0]:
                    gesture_name = "Thumb Up (Predict)"
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        # 예측 로직 (기존과 동일)
                        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        contours_canvas, _ = cv2.findContours(gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if len(contours_canvas) > 0:
                            x, y, w, h = cv2.boundingRect(np.vstack(contours_canvas))
                            padding = 20
                            x = max(0, x - padding)
                            y = max(0, y - padding)
                            w = min(canvas.shape[1] - x, w + 2 * padding)
                            h = min(canvas.shape[0] - y, h + 2 * padding)
                            
                            roi = gray_canvas[y:y+h, x:x+w]
                            roi_pil = Image.fromarray(roi)
                            max_dim = max(roi_pil.size)
                            new_img = Image.new("L", (max_dim, max_dim), 0)
                            new_img.paste(roi_pil, ((max_dim - roi_pil.width) // 2, (max_dim - roi_pil.height) // 2))
                            new_img = new_img.resize((28, 28), Image.Resampling.BICUBIC)
                            
                            with torch.no_grad():
                                if current_mode == "digit" and models_loaded["digit"]:
                                    img_tensor = transform(new_img).unsqueeze(0).to(device)
                                    output = digit_model(img_tensor)
                                    probabilities = torch.nn.functional.softmax(output, dim=1)
                                    confidence, predicted = torch.max(probabilities, 1)
                                    res_char = str(predicted.item())
                                elif current_mode == "alpha" and models_loaded["alpha"]:
                                    img_tensor = emnist_transform_fn(new_img).unsqueeze(0).to(device)
                                    output = alpha_model(img_tensor)
                                    probabilities = torch.nn.functional.softmax(output, dim=1)
                                    confidence, predicted = torch.max(probabilities, 1)
                                    res_char = alphabet_map[predicted.item()]
                                else:
                                    res_char = "Err"
                                    confidence = torch.tensor(0.0)

                            prediction_text = f"{res_char} ({confidence.item()*100:.1f}%)"
                            print(f"Predicted via Gesture: {prediction_text}")
                        last_action_time = curr_time
                    x1, y1 = 0, 0
                else:
                    # 그 외 제스처
                    x1, y1 = 0, 0

        # 캔버스와 프레임 합성
        frame = cv2.add(frame, canvas)
        
        # UI 텍스트
        mode_color = (0, 255, 0) if current_mode == "digit" else (255, 0, 255)
        cv2.Rectangle = cv2.rectangle(frame, (0,0), (width, 80), (0,0,0), -1) # 상단 블랙 바 배경
        cv2.putText(frame, f"MODE: {current_mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if prediction_text:
             cv2.putText(frame, f"Prediction: {prediction_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Hand Tracking Canvas", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            prediction_text = ""
        elif key == ord("m"):
            current_mode = "alpha" if current_mode == "digit" else "digit"
            prediction_text = ""
        elif key == ord("p"):
            pass # 키보드 예측은 유지하거나 제스처랑 병행

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
