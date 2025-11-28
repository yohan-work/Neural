import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. 디바이스 설정 (MPS -> CUDA -> CPU 순서로 확인)
# Mac의 경우 MPS(Metal Performance Shaders)를 사용하면 GPU 가속을 받을 수 있습니다.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Mac GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# 2. 데이터 준비 (MNIST Dataset)
# 2. 데이터 준비 (MNIST Dataset)
# Data Augmentation: 학습 데이터에 변형을 주어 모델의 일반화 성능 향상
train_transform = transforms.Compose([
    transforms.RandomRotation(10),      # -10도 ~ 10도 무작위 회전
    transforms.RandomAffine(0, translate=(0.1, 0.1)), # 가로/세로 10% 내외 이동
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("\n데이터셋 로드 중...")
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

# 3. CNN 모델 정의
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # Convolution Layer 1: 1 channel (gray) -> 32 channels, kernel 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        # Convolution Layer 2: 32 channels -> 64 channels, kernel 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # pool 적용 후: 14x14 -> 7x7
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5) # 과적합 방지
        self.fc2 = nn.Linear(128, 10) # 0~9 Output

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten & FC
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNIST_CNN().to(device)

# 4. 손실 함수 및 최적화 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 (Training)
epochs = 5
print(f"\n학습 시작 (Epochs: {epochs})...")
start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    model.train() # 학습 모드 설정 (Dropout 적용)
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # 데이터를 디바이스로 이동

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

end_time = time.time()
print(f'학습 완료 (소요 시간: {end_time - start_time:.2f}초)')

# 6. 모델 저장
PATH = './mnist_cnn.pth'
torch.save(model.state_dict(), PATH)
print(f"\n모델이 '{PATH}'에 저장되었습니다.")

# 7. 평가 (Evaluation)
correct = 0
total = 0
model.eval() # 평가 모드 설정 (Dropout 미적용)

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n테스트 세트 정확도: {100 * correct / total:.2f}%')

# 8. 결과 시각화
# CPU로 데이터를 다시 가져와야 시각화 가능
dataiter = iter(testloader)
images, labels = next(dataiter)
images = images[:8] # 8개만
labels = labels[:8]

# 모델 예측
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

plt.figure(figsize=(10, 4))
for i in range(8):
    ax = plt.subplot(1, 8, i + 1)
    imshow(images[i])
    ax.set_title(f'Pred: {predicted[i].item()}')
    ax.axis('off')

plt.savefig('mnist_cnn_predictions.png')
print("\n예측 결과 이미지가 'mnist_cnn_predictions.png'로 저장되었습니다.")
