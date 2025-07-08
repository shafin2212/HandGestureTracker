import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import mediapipe as mp
from mediapipe import solutions as mp_solutions
import psutil
import csv
import serial
import serial.tools.list_ports

# Worker thread for processing hand tracking
class HandTrackingWorker(QThread):
    frame_updated = pyqtSignal(np.ndarray)
    hand_landmarks_updated = pyqtSignal(list)
    gesture_detected = pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True
        self.mp_hands = mp_solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame for hand landmarks
            results = self.hands.process(frame_rgb)

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    # Extract landmarks
                    landmarks = [
                        (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                    ]
                    self.hand_landmarks_updated.emit(landmarks)

                    # Identify gesture
                    gesture = self.identify_gesture(landmarks)
                    self.gesture_detected.emit(gesture)

            # Emit the processed frame
            self.frame_updated.emit(frame)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def identify_gesture(self, landmarks):
        """
        Improved gesture detection based on finger tip and PIP comparison.
        """
        if not landmarks or len(landmarks) < 21:
            return "Unknown"

        # Define finger tip and PIP joint indices
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [2, 6, 10, 14, 18]

        fingers_open = []

        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            # If tip is above PIP (smaller y), finger is open
            fingers_open.append(tip_y < pip_y)

        if all(fingers_open):
            return "All Fingers Open"
        elif not any(fingers_open):
            return "All Fingers Closed"
        else:
            return "Unknown"

# Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sign Language Tracking")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #121212; color: white;")

        self.worker = None
        self.serial_connection = None
        self.setup_serial_connection()

        # Main layout
        main_layout = QHBoxLayout()

        # Camera Feed Section
        camera_layout = QVBoxLayout()
        self.camera_feed = QLabel(self)
        self.camera_feed.setFixedSize(800, 600)
        self.camera_feed.setAlignment(Qt.AlignCenter)
        self.camera_feed.setStyleSheet("background-color: #222; border: 2px solid #00FF00;")
        camera_layout.addWidget(self.camera_feed)

        # Buttons Layout
        button_layout = QVBoxLayout()

        # Start Tracking Button
        self.start_button = QPushButton("Start Tracking")
        self.start_button.setStyleSheet(self.get_button_style("#1de1b4"))
        self.start_button.clicked.connect(self.start_tracking)
        button_layout.addWidget(self.start_button)

        # Stop Tracking Button
        self.stop_button = QPushButton("Stop Tracking")
        self.stop_button.setStyleSheet(self.get_button_style("#FF5722"))
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        # Save Log Button
        self.log_button = QPushButton("Save Log")
        self.log_button.setStyleSheet(self.get_button_style("#9C27B0"))
        self.log_button.clicked.connect(self.save_log)
        self.log_button.setEnabled(False)
        button_layout.addWidget(self.log_button)

        camera_layout.addLayout(button_layout)
        main_layout.addLayout(camera_layout)

        # Sidebar for Features and Info
        right_layout = QVBoxLayout()

        # Detected Landmarks Table
        self.landmark_table = QTableWidget(self)
        self.landmark_table.setColumnCount(3)
        self.landmark_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.landmark_table.setStyleSheet("background-color: #333; color: white;")
        right_layout.addWidget(self.landmark_table)

        # Gesture Display
        self.gesture_label = QLabel("Detected Gesture: None", self)
        self.gesture_label.setStyleSheet("font-size: 18px; color: white;")
        right_layout.addWidget(self.gesture_label)

        # System Resource Display
        self.resource_label = QLabel("CPU: 0% | Memory: 0%")
        self.resource_label.setStyleSheet("color: white;")
        right_layout.addWidget(self.resource_label)

        # Action Information
        self.action_label = QLabel("Action: None")
        self.action_label.setStyleSheet("font-size: 18px; color: #FFEB3B;")
        right_layout.addWidget(self.action_label)

        main_layout.addLayout(right_layout)

        # Timer for Resource Monitoring
        self.resource_timer = QTimer(self)
        self.resource_timer.timeout.connect(self.update_resource_usage)
        self.resource_timer.start(1000)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def get_button_style(self, color):
        return f"""
        QPushButton {{
            background-color: {color};
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
        }}
        QPushButton:hover {{
            background-color: white;
            color: {color};
        }}
        """

    def setup_serial_connection(self):
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if "Arduino" in port.description:
                try:
                    self.serial_connection = serial.Serial(port.device, 9600, timeout=1)
                    print(f"Connected to Arduino on {port.device}")
                    break
                except Exception as e:
                    print(f"Failed to connect to Arduino: {e}")
        if not self.serial_connection:
            print("Arduino not found. Please check the connection.")

    def start_tracking(self):
        source = 0  # Webcam
        self.worker = HandTrackingWorker(source)
        self.worker.frame_updated.connect(self.update_camera_feed)
        self.worker.hand_landmarks_updated.connect(self.update_landmarks_table)
        self.worker.gesture_detected.connect(self.perform_action)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_button.setEnabled(True)

    def stop_tracking(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_button.setEnabled(False)
        self.camera_feed.clear()
        self.gesture_label.setText("Detected Gesture: None")
        self.action_label.setText("Action: None")

    def update_camera_feed(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_feed.setPixmap(pixmap)

    def update_landmarks_table(self, landmarks):
        self.landmark_table.setRowCount(len(landmarks))
        for i, (x, y, z) in enumerate(landmarks):
            self.landmark_table.setItem(i, 0, QTableWidgetItem(f"{x:.4f}"))
            self.landmark_table.setItem(i, 1, QTableWidgetItem(f"{y:.4f}"))
            self.landmark_table.setItem(i, 2, QTableWidgetItem(f"{z:.4f}"))

    def perform_action(self, gesture):
        self.gesture_label.setText(f"Detected Gesture: {gesture}")
        
        action = "None"
        if gesture == "All Fingers Open":
            action = "Turn On Light"
            self.turn_light_on()
        elif gesture == "All Fingers Closed":
            action = "Turn Off Light"
            self.turn_light_off()

        self.action_label.setText(f"Action: {action}")

    def turn_light_on(self):
        if self.serial_connection:
            self.serial_connection.write(b'1')

    def turn_light_off(self):
        if self.serial_connection:
            self.serial_connection.write(b'0')

    def save_log(self):
        log_path, _ = QFileDialog.getSaveFileName(self, "Save Landmark Log", "", "CSV Files (*.csv)")
        if log_path:
            try:
                with open(log_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["X", "Y", "Z"])
                    # No direct access to landmarks here, needs shared buffer to do it properly
                    print("Saved empty placeholder. (Landmark saving needs buffering)")
                print(f"Log saved to {log_path}")
            except Exception as e:
                print(f"Error saving log: {e}")

    def update_resource_usage(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        self.resource_label.setText(f"CPU: {cpu_usage}% | Memory: {memory_usage}%")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if self.serial_connection:
            self.serial_connection.close()
        event.accept()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
