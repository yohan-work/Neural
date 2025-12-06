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

def main():
    # 모델 로드
    model_path = './mnist_cnn.pth'
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return

    model = MNIST_CNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()

    # 전처리 변환
    transform = transforms.Compose([
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
    
    # 그리기 상태 변수
    x1, y1 = 0, 0
    prediction_text = ""

    print("Controls:")
    print(" - Show your hand to the camera")
    print(" - Draw with your INDEX FINGER")
    print(" - Press 'c' to CLEAR canvas")
    print(" - Press 'p' to PREDICT")
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
                
                # 검지 손가락 끝 (Landmark 8) 좌표 가져오기
                index_finger_tip = hand_landmarks.landmark[8]
                x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
                
                # 그리기 (검지 손가락이 펴져 있을 때만 그리는 로직을 추가할 수도 있지만, 
                # 여기서는 단순히 검지 끝을 계속 추적하여 그리게 함. 
                # 멈추려면 손을 화면 밖으로 빼거나 주먹을 쥐는 등의 제스처 로직이 필요하지만 복잡도 증가 방지를 위해 생략)
                
                cv2.circle(frame, (x, y), 10, (0, 255, 255), cv2.FILLED)
                
                if x1 == 0 and y1 == 0:
                    x1, y1 = x, y
                else:
                    # 캔버스에 그리기
                    cv2.line(canvas, (x1, y1), (x, y), (255, 255, 255), 15)
                    x1, y1 = x, y
        else:
            # 손이 감지되지 않으면 좌표 초기화 (선을 끊음)
            x1, y1 = 0, 0

        # 캔버스와 프레임 합성
        frame = cv2.add(frame, canvas)
        
        # UI 텍스트
        cv2.putText(frame, "Index Finger to Draw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "'c': Clear, 'p': Predict, 'q': Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
                
                img_tensor = transform(new_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                prediction_text = f"{predicted.item()} ({confidence.item()*100:.1f}%)"
                print(f"Predicted: {prediction_text}")
            else:
                print("Canvas is empty!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
