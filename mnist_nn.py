import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 준비 (MNIST Dataset)
# Transform: 이미지를 PyTorch Tensor로 변환하고, 정규화(Normalize)합니다.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 평균 0.5, 표준편차 0.5로 정규화 (-1 ~ 1 사이 값)
])

# 데이터셋 다운로드 및 로드
print("데이터셋 다운로드 중...")
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

# 2. 신경망 모델 정의 (Multi-Layer Perceptron)
class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        # 입력: 28x28 = 784
        self.flatten = nn.Flatten() # 2D 이미지를 1D 벡터로 펼침
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10) # 출력: 0~9까지 10개 클래스
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MNIST_MLP()

# 3. 손실 함수 및 최적화 설정
criterion = nn.CrossEntropyLoss() # 다중 분류용 손실 함수 (Softmax 포함됨)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 학습 (Training)
epochs = 5
print(f"\n학습 시작 (Epochs: {epochs})...")

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # 매 100 미니배치마다 출력
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('학습 완료')

# 5. 평가 (Evaluation)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # 가장 높은 값을 가진 인덱스가 예측값
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n테스트 세트 정확도: {100 * correct / total:.2f}%')

# 6. 결과 시각화 (일부 이미지와 예측값 확인)
dataiter = iter(testloader)
images, labels = next(dataiter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 이미지 보여주기 함수
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

# 처음 8개 이미지만 표시
plt.figure(figsize=(10, 4))
for i in range(8):
    ax = plt.subplot(1, 8, i + 1)
    imshow(images[i])
    ax.set_title(f'Pred: {predicted[i].item()}')
    ax.axis('off')

plt.savefig('mnist_predictions.png')
print("\n예측 결과 이미지가 'mnist_predictions.png'로 저장되었습니다.")
