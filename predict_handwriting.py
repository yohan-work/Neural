import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cv2

# 1. 디바이스 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 2. 모델 정의 (학습 때와 동일해야 함)
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

def predict_image(image_path, model_path='./mnist_cnn.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    # 3. 모델 로드
    model = MNIST_CNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()

    # 4. 이미지 전처리 및 숫자 분리 (Segmentation)
    try:
        # OpenCV로 이미지 읽기 (Grayscale)
        img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            print("Error: Could not read image with OpenCV.")
            return

        # 이미지 리사이즈 (너무 크면 노이즈가 많아짐, 높이 1000px로 고정)
        height, width = img_cv.shape
        if height > 1000:
            scale = 1000 / height
            new_width = int(width * scale)
            img_cv = cv2.resize(img_cv, (new_width, 1000))
        
        # 이진화 (Thresholding) - 배경과 글씨 분리
        # 배경이 밝고 글씨가 어두운 경우 (일반적인 종이) -> 반전 필요
        # 배경이 어둡고 글씨가 밝은 경우 (MNIST) -> 그대로 사용
        # 자동 판단: 평균 밝기가 127보다 크면(밝은 배경) 반전
        if np.mean(img_cv) > 127:
            img_cv = cv2.bitwise_not(img_cv)
        
        # 노이즈 제거 (Blurring)
        img_cv = cv2.GaussianBlur(img_cv, (5, 5), 0)

        # Thresholding (글씨만 확실하게 흰색으로)
        _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 윤곽선 검출 (Contours)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선 정렬 (왼쪽 -> 오른쪽)
        digit_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 노이즈 필터링: 크기가 너무 작으면 제외
            # 높이가 이미지 높이의 1/30보다 작으면 노이즈로 간주 (약 33px @ 1000px height)
            # 면적도 고려 (w * h > 500)
            if h > img_cv.shape[0] / 30 and w > 20 and (w * h) > 500:
                digit_contours.append((x, y, w, h))
        
        # x좌표 기준으로 정렬
        digit_contours.sort(key=lambda x: x[0])

        predictions = []
        processed_images = []

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        print(f"\nDetected {len(digit_contours)} digits.")

        for x, y, w, h in digit_contours:
            # 숫자 영역 추출 (여백 추가)
            padding = int(h * 0.2) # 높이의 20%만큼 패딩
            # 이미지 범위 벗어나지 않게 처리
            y_start = max(0, y - padding)
            y_end = min(img_cv.shape[0], y + h + padding)
            x_start = max(0, x - padding)
            x_end = min(img_cv.shape[1], x + w + padding)
            
            digit_img = thresh[y_start:y_end, x_start:x_end]
            
            # PIL Image로 변환
            digit_pil = Image.fromarray(digit_img)
            
            # 정사각형으로 만들기 (비율 유지)
            # 검은 배경(0)으로 패딩
            max_dim = max(digit_pil.size)
            new_img = Image.new("L", (max_dim, max_dim), 0)
            new_img.paste(digit_pil, ((max_dim - digit_pil.width) // 2, (max_dim - digit_pil.height) // 2))
            
            # Resize (28x28)
            new_img = new_img.resize((28, 28), Image.Resampling.BICUBIC)
            processed_images.append(new_img)

            # 예측
            img_tensor = transform(new_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            predictions.append((predicted.item(), confidence.item() * 100))

        # 결과 출력
        full_number = "".join([str(p[0]) for p in predictions])
        print(f"\nFinal Prediction: {full_number}")
        
        for i, (pred, conf) in enumerate(predictions):
            print(f"Digit {i+1}: {pred} ({conf:.2f}%)")

        # 시각화
        plt.figure(figsize=(12, 4))
        # 원본 이미지
        plt.subplot(1, len(predictions) + 1, 1)
        plt.imshow(cv2.imread(image_path), cmap='gray') # 원본 읽기
        plt.title(f"Original\nPred: {full_number}")
        plt.axis('off')
        
        # 분할된 숫자들
        for i, img in enumerate(processed_images):
            plt.subplot(1, len(predictions) + 1, i + 2)
            plt.imshow(img, cmap='gray')
            plt.title(f"{predictions[i][0]}\n({predictions[i][1]:.1f}%)")
            plt.axis('off')
            
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(image_path)
    else:
        print("Usage: python predict_handwriting.py <image_path>")
