"""
손글씨 수식 계산기 - Hand Tracking Calculator
MediaPipe 손 추적 + CNN 모델로 손글씨 수식을 인식하고 계산합니다.

제스처:
- ☝️  검지만: 그리기
- ✌️  검지+중지: 호버/UI 선택
- ✊  주먹: 캔버스 지우기
- 👍  엄지 올리기: 인식 & 계산
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import time
import mediapipe as mp
import re

# 1. 디바이스 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 클래스 매핑
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '=']


# 2. 모델 정의
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


def draw_toolbar(frame, selected_color):
    """간단한 툴바 그리기"""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (50, 50, 50), -1)
    
    # 색상 버튼
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    color_names = ["R", "G", "B", "Y"]
    
    for i, color in enumerate(colors):
        x1 = 140 + i * 80
        x2 = 200 + i * 80
        if selected_color == color:
            cv2.rectangle(frame, (x1-3, 7), (x2+3, 73), (255, 255, 255), 3)
        cv2.rectangle(frame, (x1, 10), (x2, 70), color, -1)
        cv2.putText(frame, color_names[i], (x1+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Clear 버튼
    cv2.rectangle(frame, (500, 10), (580, 70), (100, 100, 100), -1)
    cv2.putText(frame, "Clear", (505, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return [
        {"name": "red", "rect": (140, 10, 200, 70), "val": (0, 0, 255), "type": "color"},
        {"name": "green", "rect": (220, 10, 280, 70), "val": (0, 255, 0), "type": "color"},
        {"name": "blue", "rect": (300, 10, 360, 70), "val": (255, 0, 0), "type": "color"},
        {"name": "yellow", "rect": (380, 10, 440, 70), "val": (0, 255, 255), "type": "color"},
        {"name": "clear", "rect": (500, 10, 580, 70), "val": None, "type": "action"},
    ]


def calculate_expression(expression):
    """
    수식 문자열을 계산합니다.
    예: "3+5" -> 8, "12-4" -> 8, "6*2" -> 12, "8/2" -> 4
    """
    # '=' 제거
    expr = expression.replace('=', '').strip()
    
    if not expr:
        return None, "빈 수식"
    
    try:
        # 안전한 eval: 숫자와 기본 연산자만 허용
        # 정규식으로 유효성 검사
        if not re.match(r'^[\d+\-*/.\s]+$', expr):
            return None, "잘못된 문자"
        
        result = eval(expr)
        
        # 정수면 정수로, 소수면 소수점 2자리까지
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 2)
        
        return result, None
    except ZeroDivisionError:
        return None, "0으로 나눔"
    except Exception as e:
        return None, f"계산 오류"


def main():
    model_path = './math_symbols_cnn.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        print("먼저 train_math_symbols.py를 실행하세요.")
        return
    
    # 모델 로드
    print("모델 로드 중...")
    model = MathSymbolCNN(num_classes=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"모델 로드 완료 (Device: {device})")
    
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
    cap.set(3, 1280)
    cap.set(4, 720)
    
    canvas = None
    
    # 상태 변수
    last_action_time = 0
    cooldown = 1.0
    
    selected_color = (0, 0, 255)  # 기본 빨강
    button_hover_start = 0
    hovered_button = None
    selection_delay = 0.8
    
    x1, y1 = 0, 0
    expression_text = ""
    result_text = ""
    
    print("\n=== 손글씨 계산기 ===")
    print("제스처:")
    print(" - ☝️  검지만: 그리기")
    print(" - ✌️  검지+중지: 호버/버튼 선택")
    print(" - ✊  주먹: 캔버스 지우기")
    print(" - 👍  엄지 올리기: 인식 & 계산")
    print(" - 'q': 종료")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        if canvas is None:
            canvas = np.zeros_like(frame)
        
        # 툴바 그리기
        buttons = draw_toolbar(frame, selected_color)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 손가락 상태 체크
                fingers = []
                for id in [8, 12, 16, 20]:
                    if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id-2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # 엄지 체크
                thumb_up = (hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y and
                           hand_landmarks.landmark[4].y < hand_landmarks.landmark[8].y)
                
                cx = int(hand_landmarks.landmark[8].x * width)
                cy = int(hand_landmarks.landmark[8].y * height)
                
                # 1. 호버 (검지 + 중지) - UI 상호작용
                if fingers == [1, 1, 0, 0]:
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), 2)
                    x1, y1 = 0, 0
                    
                    hit_btn = None
                    if cy < 80:
                        for btn in buttons:
                            bx1, by1, bx2, by2 = btn["rect"]
                            if bx1 < cx < bx2 and by1 < cy < by2:
                                hit_btn = btn
                                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
                                break
                    
                    if hit_btn:
                        if hovered_button == hit_btn["name"]:
                            if time.time() - button_hover_start > selection_delay:
                                if hit_btn["type"] == "color":
                                    selected_color = hit_btn["val"]
                                    print(f"색상 선택: {hit_btn['name']}")
                                elif hit_btn["name"] == "clear":
                                    canvas = np.zeros_like(frame)
                                    expression_text = ""
                                    result_text = ""
                                    print("캔버스 지움")
                                button_hover_start = time.time()
                        else:
                            hovered_button = hit_btn["name"]
                            button_hover_start = time.time()
                    else:
                        hovered_button = None
                
                # 2. 그리기 (검지만)
                elif fingers == [1, 0, 0, 0] and cy > 80:
                    cv2.circle(frame, (cx, cy), 15, selected_color, cv2.FILLED)
                    
                    if x1 == 0 and y1 == 0:
                        x1, y1 = cx, cy
                    else:
                        cv2.line(canvas, (x1, y1), (cx, cy), selected_color, 15)
                        x1, y1 = cx, cy
                
                # 3. 주먹 - 지우기
                elif fingers == [0, 0, 0, 0] and not thumb_up:
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        canvas = np.zeros_like(frame)
                        expression_text = ""
                        result_text = ""
                        last_action_time = curr_time
                        print("캔버스 지움")
                    x1, y1 = 0, 0
                
                # 4. 엄지 올리기 - 인식 & 계산
                elif thumb_up and fingers == [0, 0, 0, 0]:
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        # 캔버스에서 문자 인식
                        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        contours_canvas, _ = cv2.findContours(
                            gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        # 후보 바운딩 박스 수집
                        candidates = []
                        for cnt in contours_canvas:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w > 10 and h > 10:
                                candidates.append((x, y, w, h))
                        
                        if candidates:
                            # 노이즈 필터링
                            areas = [w * h for x, y, w, h in candidates]
                            max_area = max(areas)
                            
                            valid_boxes = [(x, y, w, h) for i, (x, y, w, h) in enumerate(candidates)
                                          if areas[i] > max_area * 0.05]
                            
                            # X 좌표로 정렬
                            valid_boxes.sort(key=lambda b: b[0])
                            
                            results = []
                            
                            for x, y, w, h in valid_boxes:
                                padding = 20
                                x_p = max(0, x - padding)
                                y_p = max(0, y - padding)
                                w_p = min(canvas.shape[1] - x_p, w + 2 * padding)
                                h_p = min(canvas.shape[0] - y_p, h + 2 * padding)
                                
                                roi = gray_canvas[y_p:y_p+h_p, x_p:x_p+w_p]
                                roi_pil = Image.fromarray(roi)
                                
                                # 정사각형으로 만들기
                                max_dim = max(roi_pil.size)
                                new_img = Image.new("L", (max_dim, max_dim), 0)
                                new_img.paste(roi_pil, 
                                            ((max_dim - roi_pil.width) // 2,
                                             (max_dim - roi_pil.height) // 2))
                                
                                # 28x28 리사이즈
                                new_img = new_img.resize((28, 28), Image.Resampling.BICUBIC)
                                
                                # 예측
                                img_tensor = transform(new_img).unsqueeze(0).to(device)
                                
                                with torch.no_grad():
                                    output = model(img_tensor)
                                    prob = torch.nn.functional.softmax(output, dim=1)
                                    conf, pred = torch.max(prob, 1)
                                    
                                    res_char = CLASS_NAMES[pred.item()]
                                    results.append(res_char)
                                
                                # 시각적 피드백
                                cv2.rectangle(frame, (x_p, y_p), (x_p+w_p, y_p+h_p), (0, 255, 0), 2)
                            
                            expression_text = "".join(results)
                            print(f"\n인식: {expression_text}")
                            
                            # 계산
                            calc_result, error = calculate_expression(expression_text)
                            if error:
                                result_text = f"= {error}"
                            else:
                                result_text = f"= {calc_result}"
                            print(f"결과: {result_text}")
                        else:
                            expression_text = ""
                            result_text = "(빈 캔버스)"
                        
                        last_action_time = curr_time
                    x1, y1 = 0, 0
                else:
                    x1, y1 = 0, 0
        
        # 캔버스 합성
        frame = cv2.add(frame, canvas)
        
        # 정보 표시
        cv2.putText(frame, "CALCULATOR MODE", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if expression_text:
            cv2.putText(frame, f"Expression: {expression_text}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if result_text:
            cv2.putText(frame, result_text, (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imshow("Hand Tracking Calculator", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
