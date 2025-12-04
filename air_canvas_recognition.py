import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

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

def nothing(x):
    pass

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

    # 웹캠 설정
    cap = cv2.VideoCapture(0)
    
    # 트랙바 윈도우 생성 (색상 조절용)
    cv2.namedWindow("Color Detectors")
    # 초기값: 파란색 계열 (예시)
    cv2.createTrackbar("Upper Hue", "Color Detectors", 153, 180, nothing)
    cv2.createTrackbar("Upper Saturation", "Color Detectors", 255, 255, nothing)
    cv2.createTrackbar("Upper Value", "Color Detectors", 255, 255, nothing)
    cv2.createTrackbar("Lower Hue", "Color Detectors", 64, 180, nothing)
    cv2.createTrackbar("Lower Saturation", "Color Detectors", 72, 255, nothing)
    cv2.createTrackbar("Lower Value", "Color Detectors", 49, 255, nothing)

    # 그림을 그릴 캔버스 (검은 배경)
    # 초기에는 None으로 두고 프레임 크기에 맞춤
    canvas = None
    
    # 그리기 상태 변수
    x1, y1 = 0, 0
    
    # 예측 결과 텍스트
    prediction_text = ""

    print("Controls:")
    print(" - Adjust trackbars to mask your object (pen/finger)")
    print(" - Draw in the air")
    print(" - Press 'c' to CLEAR canvas")
    print(" - Press 'p' to PREDICT")
    print(" - Press 'q' to QUIT")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # 거울 모드
        
        if canvas is None:
            canvas = np.zeros_like(frame)

        # HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 트랙바 값 읽기
        u_hue = cv2.getTrackbarPos("Upper Hue", "Color Detectors")
        u_sat = cv2.getTrackbarPos("Upper Saturation", "Color Detectors")
        u_val = cv2.getTrackbarPos("Upper Value", "Color Detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color Detectors")
        l_sat = cv2.getTrackbarPos("Lower Saturation", "Color Detectors")
        l_val = cv2.getTrackbarPos("Lower Value", "Color Detectors")
        
        Upper_hsv = np.array([u_hue, u_sat, u_val])
        Lower_hsv = np.array([l_hue, l_sat, l_val])

        # 마스크 생성
        mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        
        # 노이즈 제거 (Erosion & Dilation)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        if len(contours) > 0:
            # 가장 큰 윤곽선 찾기 (가장 가까운 물체)
            cnt = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            M = cv2.moments(cnt)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # 물체가 일정 크기 이상일 때만 그리기
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                
                # 이전 점이 있으면 선 그리기
                if x1 == 0 and y1 == 0:
                    x1, y1 = center
                else:
                    # 캔버스에 그리기 (흰색 선)
                    cv2.line(canvas, (x1, y1), center, (255, 255, 255), 15)
                    # 화면에도 그리기 (사용자 피드백용)
                    # cv2.line(frame, (x1, y1), center, (0, 255, 0), 5)
                    x1, y1 = center
            else:
                x1, y1 = 0, 0
        else:
            x1, y1 = 0, 0

        # 캔버스와 프레임 합성
        # 캔버스가 흰색인 부분(그림)을 프레임에 덮어씌움 (회색으로 표시)
        frame = cv2.add(frame, canvas)
        
        # UI 텍스트 표시
        cv2.putText(frame, "Press 'c' to Clear, 'p' to Predict", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if prediction_text:
             cv2.putText(frame, f"Prediction: {prediction_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 마스크 윈도우도 보여줌 (설정 편의를 위해)
        cv2.imshow("Mask", mask)
        cv2.imshow("Air Canvas", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            prediction_text = ""
            print("Canvas Cleared")
        elif key == ord("p"):
            # 예측 수행
            # 캔버스를 그레이스케일로 변환
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            
            # ROI 찾기 (그림이 그려진 영역만 자르기 위해)
            contours_canvas, _ = cv2.findContours(gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours_canvas) > 0:
                # 모든 컨투어를 포함하는 bounding box 찾기
                x, y, w, h = cv2.boundingRect(np.vstack(contours_canvas))
                
                # 여백 추가
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(canvas.shape[1] - x, w + 2 * padding)
                h = min(canvas.shape[0] - y, h + 2 * padding)
                
                roi = gray_canvas[y:y+h, x:x+w]
                
                # PIL Image 변환
                roi_pil = Image.fromarray(roi)
                
                # 정사각형 만들기
                max_dim = max(roi_pil.size)
                new_img = Image.new("L", (max_dim, max_dim), 0)
                new_img.paste(roi_pil, ((max_dim - roi_pil.width) // 2, (max_dim - roi_pil.height) // 2))
                
                # 28x28 리사이즈
                new_img = new_img.resize((28, 28), Image.Resampling.BICUBIC)
                
                # 예측
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
