import sys
import cv2
import os
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QInputDialog
from PyQt5.QtCore import QThread, pyqtSignal
import face_recognition

# Function: Add Face to Dataset
def add_face():
    name, ok = QInputDialog.getText(None, "Input Name", "Enter Your Child's Name:")
    if not ok or not name.strip():
        print("Name input cancelled or invalid.")
        window.label.setText("Name input cancelled or invalid.")
        return

    name = name.strip()
    save_path = os.path.join("static/images/dataset", name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    max_photos = 16  # Batas maksimum jumlah foto
    window.label.setText(f"Adding face data for {name}... Press 'q' to stop or wait until 15 photos are captured.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in faces:
            face_img = frame[top:bottom, left:right]
            cv2.imwrite(f"{save_path}/{name}_{count}.jpg", face_img)
            count += 1

            # Hentikan jika jumlah foto mencapai batas
            if count >= max_photos:
                break

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_photos:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Berikan feedback pada pengguna
    if count >= max_photos:
        window.label.setText(f"Reached maximum of {max_photos} photos. Starting training...")
    else:
        window.label.setText(f"Saved {count} images for {name}.")

    # Langsung panggil fungsi untuk melatih data
    train_model()

# Background Worker Thread for Training
class TrainModelWorker(QThread):
    progress = pyqtSignal(str)

    def run(self):
        DATASET_PATH = "static/images/dataset"
        MODEL_PATH = "static/images/face_encodings.pkl"
        encodings = []
        names = []

        self.progress.emit("Training started...")

        existing_encodings = []
        existing_names = []
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as model_file:
                data = pickle.load(model_file)
                existing_encodings = data["encodings"]
                existing_names = data["names"]
            self.progress.emit(f"Loaded existing model data with {len(existing_names)} faces.")

        try:
            for name in os.listdir(DATASET_PATH):
                person_path = os.path.join(DATASET_PATH, name)
                if not os.path.isdir(person_path):
                    continue

                self.progress.emit(f"Processing {name}...")
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)

                    for encoding in face_encodings:
                        if not any(face_recognition.compare_faces(existing_encodings, encoding, tolerance=0.4)):
                            encodings.append(encoding)
                            names.append(name)

            combined_encodings = existing_encodings + encodings
            combined_names = existing_names + names

            with open(MODEL_PATH, "wb") as f:
                pickle.dump({"encodings": combined_encodings, "names": combined_names}, f)

            self.progress.emit(f"Training complete. Model saved with {len(combined_names)} faces!")

        except Exception as e:
            self.progress.emit(f"Error: {e}")

# Function: Train Model
def train_model():
    global worker
    if 'worker' in globals() and worker.isRunning():
        window.label.setText("Training is already running. Please wait...")
        return

    worker = TrainModelWorker()
    worker.progress.connect(window.label.setText)
    worker.progress.connect(print)
    worker.start()

# GUI Class
class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Software")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # Buttons
        self.add_face_btn = QPushButton("Add Face to Dataset")
        self.train_btn = QPushButton("Train Model")

        # Label
        self.label = QLabel("Choose an action:")

        # Button Actions
        self.add_face_btn.clicked.connect(add_face)
        self.train_btn.clicked.connect(train_model)

        # Add Widgets
        layout.addWidget(self.label)
        layout.addWidget(self.add_face_btn)
        layout.addWidget(self.train_btn)

        self.setLayout(layout)

# Main Function
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
