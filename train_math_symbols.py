"""
손글씨 수학 기호 인식 모델 학습
- 숫자 (0-9): 10개 클래스
- 연산자 (+, -, *, /, =): 5개 클래스
- 총 15개 클래스

데이터셋: 커스텀 생성 (MNIST + 합성 연산자 데이터)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from PIL import Image, ImageDraw, ImageFont
import random

# 1. 디바이스 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# 클래스 매핑: 0-9(숫자), 10(+), 11(-), 12(*), 13(/), 14(=)
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=']
NUM_CLASSES = len(CLASS_NAMES)

def class_to_label(class_idx):
    """클래스 인덱스를 문자로 변환"""
    return CLASS_NAMES[class_idx]

def label_to_class(label):
    """문자를 클래스 인덱스로 변환"""
    return CLASS_NAMES.index(label)


# 2. 합성 연산자 데이터 생성 클래스
class SyntheticOperatorDataset(Dataset):
    """합성 연산자 이미지 데이터셋"""
    
    def __init__(self, operator, class_idx, num_samples=2000, transform=None):
        self.operator = operator
        self.class_idx = class_idx
        self.num_samples = num_samples
        self.transform = transform
        
        # 다양한 폰트 스타일 시뮬레이션을 위한 파라미터
        self.fonts = self._get_fonts()
        
    def _get_fonts(self):
        """시스템에서 사용 가능한 폰트 찾기"""
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Times.ttc", 
            "/System/Library/Fonts/Courier.dfont",
            "/System/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        valid_fonts = []
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, 20)
                    valid_fonts.append(path)
                except:
                    pass
        if not valid_fonts:
            valid_fonts = [None]  # 기본 폰트 사용
        return valid_fonts
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 28x28 검은 배경 이미지 생성
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # 랜덤 폰트 크기 (다양성을 위해)
        font_size = random.randint(16, 24)
        
        # 폰트 로드
        font_path = random.choice(self.fonts)
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 텍스트 위치 계산 (중앙 + 약간의 랜덤 오프셋)
        # PIL 버전에 따라 textbbox 또는 textsize 사용
        try:
            bbox = draw.textbbox((0, 0), self.operator, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(self.operator, font=font)
        
        x = (28 - text_width) // 2 + random.randint(-3, 3)
        y = (28 - text_height) // 2 + random.randint(-3, 3)
        
        # 흰색으로 연산자 그리기
        draw.text((x, y), self.operator, fill=255, font=font)
        
        # 약간의 노이즈 추가
        img_array = np.array(img)
        noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.class_idx


class HandwrittenOperatorDataset(Dataset):
    """손글씨 스타일 합성 연산자 데이터셋 (더 자연스러운 버전)"""
    
    def __init__(self, operator, class_idx, num_samples=3000, transform=None):
        self.operator = operator
        self.class_idx = class_idx
        self.num_samples = num_samples
        self.transform = transform
    
    def __len__(self):
        return self.num_samples
    
    def _draw_handwritten_operator(self):
        """손글씨 스타일로 연산자 그리기"""
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = 14, 14
        
        # 랜덤 오프셋
        ox = random.randint(-2, 2)
        oy = random.randint(-2, 2)
        
        # 선 굵기
        width = random.randint(2, 4)
        
        if self.operator == '+':
            # 가로선
            h_len = random.randint(6, 10)
            draw.line([center_x - h_len + ox, center_y + oy, 
                      center_x + h_len + ox, center_y + oy], fill=255, width=width)
            # 세로선
            v_len = random.randint(6, 10)
            draw.line([center_x + ox, center_y - v_len + oy,
                      center_x + ox, center_y + v_len + oy], fill=255, width=width)
                      
        elif self.operator == '-':
            # 가로선만
            h_len = random.randint(6, 10)
            draw.line([center_x - h_len + ox, center_y + oy,
                      center_x + h_len + ox, center_y + oy], fill=255, width=width)
                      
        elif self.operator == '*':
            # X 형태 또는 점 형태
            if random.random() > 0.5:
                # X 형태
                size = random.randint(5, 8)
                draw.line([center_x - size + ox, center_y - size + oy,
                          center_x + size + ox, center_y + size + oy], fill=255, width=width)
                draw.line([center_x + size + ox, center_y - size + oy,
                          center_x - size + ox, center_y + size + oy], fill=255, width=width)
            else:
                # × 기호 (좀 더 곱하기 스타일)
                size = random.randint(5, 8)
                draw.line([center_x - size + ox, center_y - size + oy,
                          center_x + size + ox, center_y + size + oy], fill=255, width=width)
                draw.line([center_x + size + ox, center_y - size + oy,
                          center_x - size + ox, center_y + size + oy], fill=255, width=width)
                          
        elif self.operator == '/':
            # 대각선
            size = random.randint(8, 12)
            draw.line([center_x + size + ox, center_y - size + oy,
                      center_x - size + ox, center_y + size + oy], fill=255, width=width)
                      
        elif self.operator == '=':
            # 두 가로선
            h_len = random.randint(6, 10)
            gap = random.randint(3, 5)
            draw.line([center_x - h_len + ox, center_y - gap + oy,
                      center_x + h_len + ox, center_y - gap + oy], fill=255, width=width)
            draw.line([center_x - h_len + ox, center_y + gap + oy,
                      center_x + h_len + ox, center_y + gap + oy], fill=255, width=width)
        
        # 약간의 블러(가우시안) 효과 - 손글씨처럼 보이게
        img_array = np.array(img, dtype=np.float32)
        
        # 랜덤 노이즈
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def __getitem__(self, idx):
        img = self._draw_handwritten_operator()
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.class_idx


# 3. MNIST를 래핑하여 10개 클래스 유지
class MNISTWrapper(Dataset):
    """MNIST 데이터셋 래퍼 (클래스 오프셋 없이 0-9 유지)"""
    
    def __init__(self, mnist_dataset):
        self.dataset = mnist_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # label은 이미 0-9
        return img, label


# 4. 모델 정의
class MathSymbolCNN(nn.Module):
    """수학 기호 인식 CNN (15개 클래스)"""
    
    def __init__(self, num_classes=15):
        super(MathSymbolCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
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
    # 데이터 변환
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("\n[1/4] 데이터셋 준비 중...")
    
    # MNIST 데이터 로드
    print("  - MNIST 로드...")
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # MNIST 래핑 (레이블 0-9 유지)
    train_digits = MNISTWrapper(mnist_train)
    test_digits = MNISTWrapper(mnist_test)
    
    # 연산자 데이터셋 생성
    print("  - 합성 연산자 데이터 생성...")
    operators = [('+', 10), ('-', 11), ('*', 12), ('/', 13), ('=', 14)]
    
    train_operators = []
    test_operators = []
    
    for op, class_idx in operators:
        # 학습용: 폰트 기반 + 손글씨 스타일 혼합
        train_operators.append(SyntheticOperatorDataset(
            op, class_idx, num_samples=2000, transform=train_transform
        ))
        train_operators.append(HandwrittenOperatorDataset(
            op, class_idx, num_samples=4000, transform=train_transform
        ))
        
        # 테스트용
        test_operators.append(HandwrittenOperatorDataset(
            op, class_idx, num_samples=500, transform=test_transform
        ))
    
    # 전체 데이터셋 병합
    train_dataset = ConcatDataset([train_digits] + train_operators)
    test_dataset = ConcatDataset([test_digits] + test_operators)
    
    print(f"  - 학습 데이터: {len(train_dataset)} 샘플")
    print(f"  - 테스트 데이터: {len(test_dataset)} 샘플")
    
    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # 모델 생성
    print("\n[2/4] 모델 초기화...")
    model = MathSymbolCNN(num_classes=NUM_CLASSES).to(device)
    print(f"  - 클래스: {CLASS_NAMES}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 학습
    print("\n[3/4] 학습 시작...")
    epochs = 10
    start_time = time.time()
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'  [Epoch {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    end_time = time.time()
    print(f'\n학습 완료 (소요 시간: {end_time - start_time:.2f}초)')
    
    # 모델 저장
    MODEL_PATH = './math_symbols_cnn.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"모델 저장됨: {MODEL_PATH}")
    
    # 평가
    print("\n[4/4] 평가 중...")
    model.eval()
    correct = 0
    total = 0
    
    # 클래스별 정확도 계산
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    print(f'\n전체 테스트 정확도: {100 * correct / total:.2f}%')
    print("\n클래스별 정확도:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {CLASS_NAMES[i]:>3}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    # 학습 곡선 저장
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.savefig('math_symbols_training_curve.png')
    print("\n학습 곡선 저장됨: math_symbols_training_curve.png")


if __name__ == "__main__":
    main()
