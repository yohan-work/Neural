import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 데이터 준비 (XOR 문제)
# XOR는 입력이 같으면 0, 다르면 1을 출력하는 문제입니다.
# (0, 0) -> 0
# (0, 1) -> 1
# (1, 0) -> 1
# (1, 1) -> 0
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 2. 신경망 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 입력층(2개) -> 은닉층(8개)
        self.layer1 = nn.Linear(2, 8) 
        self.relu = nn.ReLU() # 활성화 함수
        # 은닉층(8개) -> 출력층(1개)
        self.layer2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid() # 0~1 사이 값으로 변환 (이진 분류)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

model = SimpleNN()

# 3. 손실 함수(Loss Function)와 최적화(Optimizer) 설정
criterion = nn.BCELoss() # Binary Cross Entropy Loss (이진 분류용)
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam Optimizer

# 4. 학습 (Training)
epochs = 10000
losses = []

print("학습 시작...")
for epoch in range(epochs):
    # 순전파 (Forward Pass)
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 역전파 (Backward Pass) 및 가중치 업데이트
    optimizer.zero_grad() # 기울기 초기화
    loss.backward()       # 기울기 계산
    optimizer.step()      # 가중치 업데이트
    
    losses.append(loss.item())
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 결과 확인 (Evaluation)
with torch.no_grad():
    predictions = model(X)
    predicted_cls = predictions.round() # 0.5 이상이면 1, 미만이면 0
    accuracy = (predicted_cls.eq(y).sum() / float(y.shape[0])).item()
    print(f'\n정확도(Accuracy): {accuracy * 100:.2f}%')
    print('예측값:\n', predictions.numpy())

# 6. 학습 과정 시각화
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_plot.png') # 그래프 저장
print("\n학습 손실 그래프가 'loss_plot.png'로 저장되었습니다.")
