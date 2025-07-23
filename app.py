from dotenv import load_dotenv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow & MediaPipe warnings

import cv2
import mediapipe as mp
import requests
from pymodbus.client import ModbusTcpClient

# ========== MODBUS TCP (OpenPLC) ========== #
modbus_client = ModbusTcpClient('127.0.0.1', port=502)  # '0.0.0.0' is invalid here
if not modbus_client.connect():
    print("❌ Could not connect to Modbus server. Is OpenPLC running?")
else:
    print("🔌 Modbus TCP Connected to OpenPLC")

# ========== AZURE OPENAI SETTINGS ========== #
load_dotenv()
AZURE_API_KEY = "OPENAI_API_KEY"
AZURE_ENDPOINT = "AZURE_ENDPOINT"
AZURE_DEPLOYMENT = "gpt-4o"
AZURE_API_VERSION = "2025-01-01-preview"

def ask_azure_gpt(finger_list):
    prompt = f"""
You are an industrial automation assistant. You are given a hand gesture in binary list format.

Use the following rules strictly:
- If the gesture is [0,1,0,0,0] → return "PLC_1_ON"
- If the gesture is [0,1,1,0,0] → return "PLC_2_ON"
- If the gesture is [0,0,0,0,0] → return "ALL_OFF"
- For any other gesture → return "NO_ACTION"

Gesture: {finger_list}
Return only the command string (e.g., PLC_1_ON).
"""

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }

    data = {
        "messages": [
            {"role": "system", "content": "You are an industrial automation assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 1
    }

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        command = result["choices"][0]["message"]["content"].strip()
        return command
    except Exception as e:
        print(f"❌ Azure GPT error: {e}")
        return "NO_ACTION"

def send_to_plc(command):
    if command == "PLC_1_ON":
        modbus_client.write_coil(0, True)
        modbus_client.write_coil(1, False)
    elif command == "PLC_2_ON":
        modbus_client.write_coil(1, True)
        modbus_client.write_coil(0, False)
    elif command == "ALL_OFF":
        modbus_client.write_coil(0, False)
        modbus_client.write_coil(1, False)

# ========== MEDIAPIPE HANDS ========== #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=1)
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# ========== START CAMERA ========== #
cap = cv2.VideoCapture(0)
print("🚀 System started. Raise gestures to control PLC via Azure GPT...")

while True:
    success, img = cap.read()
    if not success:
        print("❌ Camera not accessible.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    gesture_result = "Detecting..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        fingers = []
        if lm_list:
            fingers.append(0)  # Skip thumb
            for i in range(1, 5):
                fingers.append(1 if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1] else 0)

            # Ask GPT for command
            gesture_result = ask_azure_gpt(fingers)

            # Show on screen
            cv2.putText(img, f'Command: {gesture_result}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Send to PLC
            send_to_plc(gesture_result)

            # Log action
            print(f"➡️ Gesture: {fingers} → Command: {gesture_result}")

    cv2.imshow("Azure GPT + Hand Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
modbus_client.close()
print("✅ Modbus disconnected.")
