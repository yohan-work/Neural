import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

# 1. 디바이스 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

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
    model_path = './mnist_cnn.pth'
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please train the model first.")
        return

    # 모델 로드
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

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 화면 중앙에 인식 영역 표시 (사용자 가이드용)
        height, width = frame.shape[:2]
        
        # 이미지 전처리
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 이진화 (Adaptive Thresholding이 조명 변화에 더 강함)
        # 조명 환경에 따라 thresh 값을 조정해야 할 수도 있음
        # 여기서는 Otsu's binarization 사용
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 윤곽선 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # 너무 작거나 너무 큰 영역 제외
            if area > 1000 and area < 50000:
                # 비율 체크 (숫자는 보통 세로로 길거나 정사각형에 가까움)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 1.5:
                    digit_contours.append((x, y, w, h))

        # 인식된 각 영역에 대해 예측 수행
        for x, y, w, h in digit_contours:
            # ROI 추출 및 패딩
            padding = int(h * 0.2)
            y_start = max(0, y - padding)
            y_end = min(height, y + h + padding)
            x_start = max(0, x - padding)
            x_end = min(width, x + w + padding)
            
            roi = thresh[y_start:y_end, x_start:x_end]
            
            if roi.size == 0:
                continue

            # PIL Image로 변환하여 모델 입력 형태로 맞춤
            roi_pil = Image.fromarray(roi)
            
            # 정사각형 만들기
            max_dim = max(roi_pil.size)
            new_img = Image.new("L", (max_dim, max_dim), 0)
            new_img.paste(roi_pil, ((max_dim - roi_pil.width) // 2, (max_dim - roi_pil.height) // 2))
            
            # 28x28 리사이즈
            new_img = new_img.resize((28, 28), Image.Resampling.BICUBIC)
            
            # 텐서 변환 및 예측
            img_tensor = transform(new_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 신뢰도가 일정 수준 이상일 때만 표시
            if confidence.item() > 0.6:
                # 박스 그리기
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 텍스트 표시
                label = f"{predicted.item()} ({confidence.item()*100:.1f}%)"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 결과 화면 출력
        cv2.imshow('Real-time Digit Recognition', frame)
        # 디버깅용: 전처리된 화면도 보고 싶다면 아래 주석 해제
        # cv2.imshow('Threshold', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
