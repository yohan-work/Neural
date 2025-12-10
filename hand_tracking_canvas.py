import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
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
    
    # 상태 변수
    x1, y1 = 0, 0
    prediction_text = ""
    current_mode = "digit" # 'digit' or 'alpha'
    
    # 알파벳 매핑 (1-26 -> A-Z, but model outputs 0-25)
    alphabet_map = {i: chr(65+i) for i in range(26)}

    print("Controls:")
    print(" - Show your hand to the camera")
    print(" - Draw with your INDEX FINGER")
    print(" - Press 'c' to CLEAR canvas")
    print(" - Press 'p' to PREDICT")
    print(" - Press 'm' to SWITCH MODE (Digit <-> Alphabet)")
    print(" - Press 'q' to QUIT")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # 거울 모드
        height, width, _ = frame.shape
        
        if canvas is None:
            canvas = np.zeros_like(frame)

        # MediaPipe 처리를 위해 BGR -> RGB 변환
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        # 손이 감지되면
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 손 뼈대 그리기
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 검지 손가락 끝 (Landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
                
                cv2.circle(frame, (x, y), 10, (0, 255, 255), cv2.FILLED)
                
                if x1 == 0 and y1 == 0:
                    x1, y1 = x, y
                else:
                    # 캔버스에 그리기
                    cv2.line(canvas, (x1, y1), (x, y), (255, 255, 255), 15)
                    x1, y1 = x, y
        else:
            x1, y1 = 0, 0

        # 캔버스와 프레임 합성
        frame = cv2.add(frame, canvas)
        
        # UI 텍스트
        mode_color = (0, 255, 0) if current_mode == "digit" else (255, 0, 255)
        cv2.putText(frame, f"MODE: {current_mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)
        cv2.putText(frame, "'c': Clear, 'p': Predict, 'm': Switch", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if prediction_text:
             cv2.putText(frame, f"Prediction: {prediction_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Hand Tracking Canvas", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            prediction_text = ""
            print("Canvas Cleared")
        elif key == ord("m"):
            current_mode = "alpha" if current_mode == "digit" else "digit"
            prediction_text = ""
            print(f"Switched to {current_mode} mode")
        elif key == ord("p"):
            # 예측 수행
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
                        # EMNIST 전처리 사용
                        # EMNIST 모델은 학습 시 회전된 이미지를 학습했으므로, 
                        # 우리가 그린 정방향 이미지를 모델에 넣기 전에 똑같이 회전시켜줘야 함.
                        img_tensor = emnist_transform_fn(new_img).unsqueeze(0).to(device)
                        output = alpha_model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        res_char = alphabet_map[predicted.item()]
                    else:
                        res_char = "Err"
                        confidence = torch.tensor(0.0)

                prediction_text = f"{res_char} ({confidence.item()*100:.1f}%)"
                print(f"Predicted: {prediction_text}")
            else:
                print("Canvas is empty!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
