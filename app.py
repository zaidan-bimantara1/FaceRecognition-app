from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
app = Flask(__name__)

# Halaman Login
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip().lower()  # Normalize input
        if username == "user":
            return redirect(url_for("user_home"))
        elif username == "admin":
            return redirect(url_for("admin_home"))
        else:
            return render_template("login.html", error="Invalid username!")
    return render_template("login.html")

# Halaman Beranda User
@app.route("/user_home")
def user_home():
    return render_template("face_recognition.html")

# Halaman Beranda Admin
@app.route("/admin_home")
def admin_home():
    return render_template("admin_home.html")

# Route untuk menjalankan Face Recognition GUI
import sys
import os
import logging

logging.basicConfig(level=logging.DEBUG)

@app.route("/run_face_recognition")
def run_face_recognition():
    try:
        python_path = sys.executable  # Mendapatkan path interpreter Python saat ini
        script_path = os.path.join(os.getcwd(), "main.py")  # Path ke main.py
        logging.debug(f"Using Python: {python_path}")
        logging.debug(f"Running script: {script_path}")
        subprocess.Popen([python_path, script_path])
        # Redirect otomatis ke halaman sebelumnya
        return redirect(url_for("user_home"))  # Sesuaikan dengan halaman yang Anda tuju setelah ini
    except Exception as e:
        return f"Failed to run Face Recognition: {str(e)}", 500

# Route untuk Train Data
logging.basicConfig(level=logging.DEBUG)

@app.route("/train_data_action")
def train_data_action():
    try:
        python_path = sys.executable  # Mendapatkan path interpreter Python saat ini
        script_path = os.path.join(os.getcwd(), "train_data.py")  # Path ke train_data.py
        logging.debug(f"Using Python: {python_path}")
        logging.debug(f"Running script: {script_path}")
        subprocess.Popen([python_path, script_path])
        # Redirect otomatis ke halaman sebelumnya
        return redirect(url_for("admin_home"))  # Sesuaikan dengan halaman yang Anda tuju setelah ini
    except Exception as e:
        return f"Failed to run Face Recognition: {str(e)}", 500

#Route untuk real time detection
logging.basicConfig(level=logging.DEBUG)

@app.route("/real_time")
def real_time():
    try:
        python_path = sys.executable  # Mendapatkan path interpreter Python saat ini
        script_path = os.path.join(os.getcwd(), "real_time.py")  # Path ke train_data.py
        logging.debug(f"Using Python: {python_path}")
        logging.debug(f"Running script: {script_path}")
        subprocess.Popen([python_path, script_path])
        # Redirect otomatis ke halaman sebelumnya
        return redirect(url_for("user_home"))  # Sesuaikan dengan halaman yang Anda tuju setelah ini
    except Exception as e:
        return f"Failed to run Face Recognition: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)

from flaskwebgui import FlaskUI
from app import app  # Pastikan Flask Anda didefinisikan di file ini

# Buat instance FlaskUI
ui = FlaskUI(app, width=800, height=600)

if __name__ == "__main__":
    run_flask()

