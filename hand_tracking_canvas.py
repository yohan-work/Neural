import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import time
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

def draw_toolbar(frame, selected_color, current_tool):
    # Toolbar Background
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (50, 50, 50), -1)
    
    # Define Buttons
    # Colors: Red, Green, Blue, Yellow
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR
    color_names = ["Red", "Green", "Blue", "Yellow"]
    
    # 1. Color Buttons (Left side)
    for i, color in enumerate(colors):
        x1 = 140 + i * 90
        x2 = 210 + i * 90
        y1 = 10
        y2 = 70
        
        # Highlight selected
        if current_tool == "draw" and selected_color == color:
             cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 3)
             
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, color_names[i][0], (x1+25, y1+45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 2. Tools: Eraser, Save (Right side)
    # Eraser
    ex1, ex2 = 520, 590
    ey1, ey2 = 10, 70
    if current_tool == "eraser":
         cv2.rectangle(frame, (ex1-3, ey1-3), (ex2+3, ey2+3), (255, 255, 255), 3)
    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 0), -1)
    cv2.putText(frame, "Ers", (ex1+10, ey1+45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save
    sx1, sx2 = 610, 680
    sy1, sy2 = 10, 70
    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (100, 100, 100), -1)
    cv2.putText(frame, "Save", (sx1+5, sy1+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return [
        {"name": "red", "rect": (140, 10, 210, 70), "val": (0, 0, 255), "type": "color"},
        {"name": "green", "rect": (230, 10, 300, 70), "val": (0, 255, 0), "type": "color"},
        {"name": "blue", "rect": (320, 10, 390, 70), "val": (255, 0, 0), "type": "color"},
        {"name": "yellow", "rect": (410, 10, 480, 70), "val": (0, 255, 255), "type": "color"},
        {"name": "eraser", "rect": (520, 10, 590, 70), "val": None, "type": "tool"},
        {"name": "save", "rect": (610, 10, 680, 70), "val": None, "type": "action"}
    ]

def main():
    # 모델 로드 (Digit & Alphabet)
    digit_model_path = './mnist_cnn.pth'
    alpha_model_path = './emnist_cnn.pth'
    
    digit_model = MNIST_CNN().to(device)
    alpha_model = EMNIST_CNN().to(device)
    
    models_loaded = {"digit": False, "alpha": False}

    if os.path.exists(digit_model_path):
        try:
            digit_model.load_state_dict(torch.load(digit_model_path, map_location=device))
            digit_model.eval()
            models_loaded["digit"] = True
            print("Digit model loaded.")
        except:
            print("Failed to load digit model.")
    
    if os.path.exists(alpha_model_path):
        try:
            alpha_model.load_state_dict(torch.load(alpha_model_path, map_location=device))
            alpha_model.eval()
            models_loaded["alpha"] = True
            print("Alphabet model loaded.")
        except:
             print("Failed to load alphabet model.")

    if not models_loaded["digit"] and not models_loaded["alpha"]:
        print("Error: No models found. Please train at least one model.")
        return

    # 전처리 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    emnist_transform_fn = transforms.Compose([
        transforms.functional.hflip,
        lambda x: transforms.functional.rotate(x, -90),
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
    # 해상도를 높여서 툴바 공간 확보
    cap.set(3, 1280)
    cap.set(4, 720)
    
    # 캔버스 초기화
    canvas = None
    
    # 제스처 쿨다운 및 상태 변수
    last_action_time = 0
    cooldown = 1.0 
    
    # UI Interaction State
    selected_color = (0, 0, 255) # Default Red (BGR)
    current_tool = "draw" # 'draw' or 'eraser'
    button_hover_start = 0
    hovered_button = None
    selection_delay = 0.8 # Seconds to hold to select
    
    x1, y1 = 0, 0
    prediction_text = ""
    current_mode = "digit" 
    alphabet_map = {i: chr(65+i) for i in range(26)}

    print("Controls:")
    print(" - Toolbar: Hover to select Color/Eraser/Save")
    print(" - ☝️  Index Up: DRAW")
    print(" - ✌️  Index + Middle Up: HOVER (Move cursor)")
    print(" - ✊  Fist: CLEAR ALL")
    print(" - 👍  Thumb Up: PREDICT")
    print(" - 🤘  Rock: SWITCH MODE (Digit/Alpha)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) 
        height, width, _ = frame.shape
        
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Draw Toolbar
        buttons = draw_toolbar(frame, selected_color, current_tool)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        gesture_name = "None"
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Fingers Check
                fingers = []
                
                # 4 Fingers (Index~Pinky)
                # If tip y < pip y -> Open (1)
                for id in [8, 12, 16, 20]:
                    if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id-2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                        
                # Thumb Check (Simple y-check for "Thumb Up" gesture)
                thumb_up = False
                if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y and \
                   hand_landmarks.landmark[4].y < hand_landmarks.landmark[8].y:
                       thumb_up = True

                cx, cy = int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)

                # Logic
                
                # 1. Hover (Index + Middle) -> UI Interaction
                if fingers == [1, 1, 0, 0]:
                    gesture_name = "Hover / Select"
                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), 2) 
                    x1, y1 = 0, 0
                    
                    # Check button collision
                    hit_btn = None
                    if cy < 80: # Toolbar Area
                        for btn in buttons:
                            bx1, by1, bx2, by2 = btn["rect"]
                            if bx1 < cx < bx2 and by1 < cy < by2:
                                hit_btn = btn
                                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2) # Highlight hover
                                break
                    
                    if hit_btn:
                        if hovered_button == hit_btn["name"]:
                            if time.time() - button_hover_start > selection_delay:
                                # ACTION!
                                if hit_btn["type"] == "color":
                                    selected_color = hit_btn["val"]
                                    current_tool = "draw"
                                    print(f"Selected Color: {hit_btn['name']}")
                                elif hit_btn["type"] == "tool":
                                    if hit_btn["name"] == "eraser":
                                        current_tool = "eraser"
                                        print("Selected Eraser")
                                elif hit_btn["type"] == "action":
                                    if hit_btn["name"] == "save":
                                        # Save logic
                                        ts = int(time.time())
                                        filename = f"artwork_{ts}.png"
                                        # Combine canvas and frame (optional, or just canvas)
                                        # Usually artwork means the drawing itself
                                        # Let's save the canvas on black background, or inverse? 
                                        # Let's save the canvas as is (black bg, colored lines)
                                        cv2.imwrite(filename, canvas)
                                        cv2.putText(frame, "Saved!", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        print(f"Saved to {filename}")
                                        # Add small delay to prevent multiple saves
                                        time.sleep(0.5)
                                        
                                button_hover_start = time.time() # Reset to avoid rapid fire
                        else:
                            hovered_button = hit_btn["name"]
                            button_hover_start = time.time()
                    else:
                        hovered_button = None

                # 2. Draw (Index Only)
                elif fingers == [1, 0, 0, 0] and cy > 80: # Canvas Area
                    gesture_name = "Draw"
                    draw_color = selected_color if current_tool == "draw" else (0, 0, 0)
                    thickness = 15 if current_tool == "draw" else 50
                    
                    cv2.circle(frame, (cx, cy), 15, draw_color, cv2.FILLED)
                    
                    if x1 == 0 and y1 == 0:
                        x1, y1 = cx, cy
                    else:
                        cv2.line(canvas, (x1, y1), (cx, cy), draw_color, thickness)
                        x1, y1 = cx, cy

                # 3. Fist -> Clear
                elif fingers == [0, 0, 0, 0] and not thumb_up:
                    gesture_name = "Fist (Clear)"
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        canvas = np.zeros_like(frame)
                        prediction_text = ""
                        last_action_time = curr_time
                        print("Canvas Cleared")
                    x1, y1 = 0, 0

                # 4. Rock -> Switch Mode
                elif fingers == [1, 0, 0, 1]:
                    gesture_name = "Rock (Switch)"
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        current_mode = "alpha" if current_mode == "digit" else "digit"
                        prediction_text = ""
                        last_action_time = curr_time
                        print(f"Mode: {current_mode}")
                    x1, y1 = 0, 0
                
                # 5. Thumb Up -> Predict
                elif thumb_up and fingers == [0, 0, 0, 0]:
                    gesture_name = "Predict"
                    curr_time = time.time()
                    if curr_time - last_action_time > cooldown:
                        # Prediction Logic (Same as before)
                        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        contours_canvas, _ = cv2.findContours(gray_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if len(contours_canvas) > 0:
                            x, y, w, h = cv2.boundingRect(np.vstack(contours_canvas))
                            padding = 20
                            x = max(0, x - padding); y = max(0, y - padding)
                            w = min(canvas.shape[1] - x, w + 2 * padding)
                            h = min(canvas.shape[0] - y, h + 2 * padding)
                            
                            roi = gray_canvas[y:y+h, x:x+w]
                            roi_pil = Image.fromarray(roi)
                            max_dim = max(roi_pil.size)
                            new_img = Image.new("L", (max_dim, max_dim), 0)
                            new_img.paste(roi_pil, ((max_dim - roi_pil.width) // 2, (max_dim - roi_pil.height) // 2))
                            
                            # Prediction
                            with torch.no_grad():
                                if current_mode == "digit" and models_loaded["digit"]:
                                    new_img_resized = new_img.resize((28, 28), Image.Resampling.BICUBIC)
                                    img_tensor = transform(new_img_resized).unsqueeze(0).to(device)
                                    output = digit_model(img_tensor)
                                    prob = torch.nn.functional.softmax(output, dim=1)
                                    conf, pred = torch.max(prob, 1)
                                    res_char = str(pred.item())
                                elif current_mode == "alpha" and models_loaded["alpha"]:
                                    new_img_resized = new_img.resize((28, 28), Image.Resampling.BICUBIC)
                                    img_tensor = emnist_transform_fn(new_img_resized).unsqueeze(0).to(device)
                                    output = alpha_model(img_tensor)
                                    prob = torch.nn.functional.softmax(output, dim=1)
                                    conf, pred = torch.max(prob, 1)
                                    res_char = alphabet_map[pred.item()]
                                else:
                                    res_char = "Err"
                                    conf = torch.tensor(0.0)

                            prediction_text = f"{res_char} ({conf.item()*100:.1f}%)"
                            print(f"Prediction: {prediction_text}")
                        last_action_time = curr_time
                    x1, y1 = 0, 0
                else:
                    x1, y1 = 0, 0

        # Merge Canvas
        # For Eraser to work effectively on visual frame, we might need to mask it properly
        # But here 'black' color on Add operation generally does nothing if bg is real frame.
        # But we are doing add(frame, canvas). Black on canvas (0,0,0) is transparent.
        # So 'Eraser' drawing black lines on canvas effectively removes the colored lines from the canvas layer.
        # That works for 'removing drawings'.
        
        # However, to see the 'eraser cursor' track, we need to handle it.
        # It's handled by drawing black on canvas.
        
        frame = cv2.add(frame, canvas)
        
        # UI Information
        cv2.putText(frame, f"MODE: {current_mode.upper()}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tool: {current_tool.upper()}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, selected_color, 2)
        if prediction_text:
             cv2.putText(frame, f"Pred: {prediction_text}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Hand Tracking Canvas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            prediction_text = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
