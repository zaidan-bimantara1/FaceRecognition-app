import sys
import cv2
import os
import pickle
import face_recognition
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

# Background Worker for Face Recognition
class FaceRecognitionWorker(QThread):
    frame_signal = pyqtSignal(object)  # Send camera frame to UI
    result_signal = pyqtSignal(str)   # Send detection result

    def run(self):
        MODEL_PATH = "static/images/face_encodings.pkl"
        try:
            with open(MODEL_PATH, "rb") as model_file:
                data = pickle.load(model_file)
                known_encodings = data["encodings"]
                known_names = data["names"]
        except FileNotFoundError:
            self.result_signal.emit("Model not trained!")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.result_signal.emit("Camera not detected!")
            return

        detected_name = "Unknown"

        # Capture one frame
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_signal.emit(rgb_frame)  # Send frame to UI for display

            # Process face detection
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                if True in matches:
                    match_index = matches.index(True)
                    detected_name = known_names[match_index]

        cap.release()
        self.result_signal.emit(detected_name)

# Main Application Class
class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Scanner")
        self.setGeometry(200, 200, 800, 600)  # Adjust window size
        self.detected_name = "Unknown"  # Initialize detected_name attribute
        self.initUI()

    def initUI(self):
        # Layout
        self.layout = QVBoxLayout()
        self.label = QLabel("Press Start to Scan")
        self.label.setStyleSheet("font-size: 16px;")
        self.layout.addWidget(self.label)

        # Start Button
        self.start_button = QPushButton("Start Face Recognition")
        self.start_button.clicked.connect(self.start_face_recognition)
        self.layout.addWidget(self.start_button)

        # Camera Feed Area with Frame
        self.camera_container = QLabel(self)
        self.camera_container.setStyleSheet("border: 5px solid black;")  # Add black frame
        self.camera_container.setFixedSize(800, 500)

        self.camera_label = QLabel(self.camera_container)
        self.camera_label.setGeometry(5, 5, 790, 490)  # Fit inside the frame
        self.layout.addWidget(self.camera_container)

        # Scan Line
        self.scan_line = QLabel(self.camera_container)
        # Pastikan warna merah diterapkan
        self.scan_line.setStyleSheet("background-color: #FF0000;")
        self.scan_line.setFixedHeight(3)
        self.scan_line.setFixedWidth(self.camera_label.width())
        self.scan_line.hide()  # Hide initially

        self.setLayout(self.layout)

        # Timer for moving the scan line
        self.timer = QTimer()
        self.timer.timeout.connect(self.move_scan_line)
        self.scan_duration = 5000  # Animation duration in milliseconds
        self.elapsed_time = 0
        self.direction = 1  # 1 for down, -1 for up

    def start_face_recognition(self):
        self.label.setText("Processing...")
        self.start_button.setEnabled(False)  # Disable button during processing

        self.worker = FaceRecognitionWorker()
        self.worker.frame_signal.connect(self.update_camera_feed)
        self.worker.result_signal.connect(self.store_detected_name)
        self.worker.start()

    def update_camera_feed(self, frame):
        # Convert frame to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Display frame in QLabel with scaling
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size())  # Scale to fit label size
        self.camera_label.setPixmap(scaled_pixmap)

        # Start animation after frame is displayed
        self.scan_line.show()
        self.elapsed_time = 0  # Reset elapsed time
        self.timer.start(50)  # Move scan line every 50ms

    def move_scan_line(self):
        # Calculate elapsed time
        self.elapsed_time += self.timer.interval()
        if self.elapsed_time >= self.scan_duration:
            self.timer.stop()  # Stop animation after duration
            self.scan_line.hide()
            self.show_result()
            return

        # Move the red scan line up and down inside the frame
        current_y = self.scan_line.y()
        if self.direction == 1 and current_y >= self.camera_label.height() - 3:
            self.direction = -1  # Reverse to up
        elif self.direction == -1 and current_y <= 0:
            self.direction = 1  # Reverse to down

        self.scan_line.move(5, current_y + (10 * self.direction))  # Adjusted for container padding

    def store_detected_name(self, name):
        self.detected_name = name  # Store detected name

    def show_result(self):
        # Display detection result after animation ends
        self.label.setText(f"Detected: {self.detected_name}")
        self.start_button.setEnabled(True)  # Re-enable button

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
