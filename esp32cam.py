import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from pathlib import Path
import requests
import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from collections import deque, Counter
from sklearn.preprocessing import StandardScaler
import time

# ---- Paths ----
BASE_DRIVE = Path.home() / "Desktop" / "Mechatronics_Fundamentals_Project" / "HandMimic_Dataset_Local"
ENCODER_PATH = BASE_DRIVE / "label_encoder.pkl"
SCALER_PATH = BASE_DRIVE / "scaler.pkl"
MODEL_PATH = BASE_DRIVE / "gesture_model.keras"

print("Model path:", MODEL_PATH)

# ---- Load model and preprocessing ----
print("Loading model and preprocessing files...")
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model and preprocessing files loaded successfully.")

# ---- Mediapipe ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


# ---- Parameters ----
WINDOW_SIZE = 5
MIN_HAND_DETECTED = 3
PREDICT_HISTORY_SIZE = 20

# ---- Sliding buffers ----
frames = deque(maxlen=WINDOW_SIZE)
predict_list = deque(maxlen=PREDICT_HISTORY_SIZE)
last_action = None

# ---- Feature Extraction ----
FEATURE_COLS = [
    # Static features
    "thumb_angle",
    "index_angle",
    "middle_angle",
    "ring_angle",
    "pinky_angle",
    "thumb_index_distance",
    "hand_center_x",
    "hand_center_y",
    "hand_center_z",
    "hand_width",
    "hand_height",
    "num_fingers_up",
    # Dynamic features
    "move_dx",
    "move_dy",
    "move_dz",
    "move_distance",
    "move_ratio_xy",
    "move_angle_xy",
    "move_angle_yz"
]

# ---- Request Image from ESP32 ----
ESP32_IP = "192.168.4.1"

def get_image_from_esp32():
    try:
        response = requests.get('http://{ESP32_IP}/capture/')
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

# ---- Feature Extraction ----
def extract_features(frames):
    features = {f: np.nan for f in FEATURE_COLS}
    hand_flags = [f["has_hand"] for f in frames]
    if sum(hand_flags) < MIN_HAND_DETECTED:
        return None

    lm_data = np.array([f["landmarks"] for f in frames if f["has_hand"]])
    mid = lm_data[len(lm_data)//2]

    # Center
    center = mid.mean(axis=0)
    features["hand_center_x"], features["hand_center_y"], features["hand_center_z"] = center

    # Thumb angle
    thumb_vec = mid[4] - mid[0]
    features["thumb_angle"] = np.degrees(np.arctan2(thumb_vec[1], thumb_vec[0]))

    # Distances and angles
    features["thumb_index_distance"] = np.linalg.norm(mid[4] - mid[8])
    def f_angle(a, b): return np.degrees(np.arctan2(*(mid[b] - mid[a])[1::-1]))
    features["index_angle"] = f_angle(5, 8)
    features["middle_angle"] = f_angle(9, 12)
    features["ring_angle"] = f_angle(13, 16)
    features["pinky_angle"] = f_angle(17, 20)

    # Size
    features["hand_width"] = mid[:, 0].max() - mid[:, 0].min()
    features["hand_height"] = mid[:, 1].max() - mid[:, 1].min()

    # Fingers up
    count_up = 0
    for f_idx, pip_idx in zip([4, 8, 12, 16, 20], [2, 6, 10, 14, 18]):
        if mid[f_idx, 1] < mid[pip_idx, 1]:
            count_up += 1
    features["num_fingers_up"] = count_up

    # Motion
    first, last = lm_data[0].mean(axis=0), lm_data[-1].mean(axis=0)
    move = last - first
    features["move_dx"], features["move_dy"], features["move_dz"] = move
    features["move_distance"] = np.linalg.norm(move)
    features["move_ratio_xy"] = abs(move[0]) / (abs(move[1]) + 1e-6)
    features["move_angle_xy"] = np.degrees(np.arctan2(move[1], move[0]))
    features["move_angle_yz"] = np.degrees(np.arctan2(move[2], move[1]))

    return np.array([features[f] for f in FEATURE_COLS])

# ---- Main Loop ----
while True:
    time.sleep(0.1)
    frame = get_image_from_esp32()
    if frame is None:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    frame_data = {"has_hand": False, "landmarks": None}
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        lm_array = np.array([[p.x, p.y, p.z] for p in lm.landmark])
        frame_data["has_hand"] = True
        frame_data["landmarks"] = lm_array

    frames.append(frame_data)

    if len(frames) == WINDOW_SIZE:
        feats = extract_features(list(frames))
        if feats is None:
            action = "No Hand Detected"
        else:
            X = scaler.transform([feats])
            probs = model.predict(X, verbose=0)[0]
            action = label_encoder.inverse_transform([np.argmax(probs)])[0]

        predict_list.append(action)

        most_common = Counter(predict_list).most_common(1)[0][0]
        if most_common != last_action:
            last_action = most_common
            print("👉 Action detected:", last_action)

            print(f"Sending result '{last_action}' to ESP32...")
            result_payload = {'action': last_action}
            response = requests.post(f'http://{ESP32_IP}/set_action/', json=result_payload)
            if response.status_code == 200:
                print(f"Action sent to ESP32: {last_action}")
            else:
                print(f"Failed to send action to ESP32. Status code: {response.status_code}")
