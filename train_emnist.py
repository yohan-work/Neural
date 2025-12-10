import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import ssl

# SSL 인증서 오류 해결
ssl._create_default_https_context = ssl._create_unverified_context

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

# 2. 데이터 준비 (EMNIST Letters)
# EMNIST는 이미지가 90도 회전되어 있고 뒤집혀 있을 수 있어 확인 및 변환이 필요합니다.
# 하지만 보통 torchvision 로드 시 전처리를 해주면 됩니다.
# 여기서는 사용자 경험상 바로 사용하기 위해 회전/대칭 변환 함수를 추가합니다.

def emnist_transform(img):
    # EMNIST 이미지는 보통 회전되어 있음. 
    # PIL Image -> Tensor로 변환 후, 다시 PIL로 바꿔서 돌리거나 
    # Tensor 상태에서 처리. torchvision EMNIST는 (Height, Width)가 뒤집혀서 로드되는 경우가 많음.
    return lambda x: transforms.functional.rotate(transforms.functional.hflip(x), -90)

# 학습용 변환
train_transform = transforms.Compose([
    lambda x: transforms.functional.rotate(transforms.functional.hflip(x), -90), # EMNIST 맞춤 변환
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 테스트용 변환
test_transform = transforms.Compose([
    lambda x: transforms.functional.rotate(transforms.functional.hflip(x), -90),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Patch EMNIST URL
torchvision.datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

print("\nEMNIST 데이터셋 로드 중... (시간이 조금 걸릴 수 있습니다)")
try:
    # 'letters' split: 26 classes (A-Z)
    # Class labels are 1-based (1=A, ..., 26=Z), but PyTorch dataset maps them to 0-based if not careful, 
    # OR we handle the 1-based index (0 is reserved for N/A in the original mapping).
    # torchvision EMNIST 'letters' returns labels 1-26. We need to map them to 0-25 for CrossEntropyLoss usually, 
    # but let's check. Actually, 'letters' might be 1-26.
    # To be safe, we will subtract 1 from target transform if they are 1-based.
    
    trainset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True,
                                            download=True, transform=train_transform,
                                            target_transform=lambda x: x - 1) # 1-26 -> 0-25
    
    testset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False,
                                           download=True, transform=test_transform,
                                           target_transform=lambda x: x - 1)
                                           
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)
except Exception as e:
    print(f"Error loading EMNIST: {e}")
    exit()

# 3. 모델 정의
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

model = EMNIST_CNN().to(device)

# 4. 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
epochs = 5 # EMNIST는 데이터가 많아서 5 Epoch 정도면 충분할 수 있음 (시간 단축)
print(f"\n학습 시작 (Epochs: {epochs})...")
start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    epoch_loss = running_loss / len(trainloader)
    print(f'[Epoch {epoch + 1}] Loss: {epoch_loss:.4f}')

end_time = time.time()
print(f'학습 완료 (소요 시간: {end_time - start_time:.2f}초)')

# 6. 저장
PATH = './emnist_cnn.pth'
torch.save(model.state_dict(), PATH)
print(f"\n모델이 '{PATH}'에 저장되었습니다.")

# 7. 검증
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n테스트 세트 정확도: {100 * correct / total:.2f}%')
