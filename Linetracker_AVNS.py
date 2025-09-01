import cv2
import requests
import numpy as np
import time
import serial

# === CONFIGURATION ===
ESP32_IP = "http://192.168.252.13"  # Replace with your ESP32-CAM IP
STREAM_URL = f"{ESP32_IP}:81/stream"
SERIAL_PORT = "COM6"  # Replace with your Arduino COM port
BAUD_RATE = 9600

# === SETUP SERIAL COMMUNICATION ===
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on {SERIAL_PORT}")
except Exception as e:
    print(f"Serial connection failed: {e}")
    ser = None

# === SETUP VIDEO STREAM ===
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("Failed to connect to ESP32-CAM stream.")
    exit()

# === MAIN LOOP ===
last_sent = ''
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    height, width, _ = frame.shape
    frame_center = width // 2

    # Crop bottom part of the image (region of interest)
    roi = frame[height // 2:height, 0:width]

    # Grayscale + blur + adaptive threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cx = frame_center
    cy = height - 30  # Visual marker

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + height // 2
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"x={cx}", (cx - 40, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === DECISION & SEND COMMAND ===
        if cx < frame_center - 20:
            cmd = 'L'
            decision = "LEFT"
        elif cx > frame_center + 20:
            cmd = 'R'
            decision = "RIGHT"
        else:
            cmd = 'F'
            decision = "FORWARD"

        # Only send if changed
        if cmd != last_sent and ser:
            ser.write(cmd.encode())
            print(f"Sent command: {cmd}")
            last_sent = cmd
    else:
        cmd = 'S'
        decision = "STOP (No line)"
        if cmd != last_sent and ser:
            ser.write(cmd.encode())
            print("Sent command: S")
            last_sent = cmd

    # Draw visual aids
    cv2.line(frame, (frame_center, 0), (frame_center, height), (255, 255, 0), 1)
    cv2.putText(frame, "CENTER", (frame_center - 40, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(frame, decision, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2)

    # Show window
    frame_resized = cv2.resize(frame, (800, 600))
    cv2.imshow("ESP32 Line Follower", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
