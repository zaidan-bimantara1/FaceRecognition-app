import subprocess
import os

def start_flask():
    # Jalankan Flask di background
    flask_process = subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    print("Flask server is running...")
    return flask_process

if __name__ == "__main__":
    flask_process = start_flask()
    try:
        flask_process.communicate()
    except KeyboardInterrupt:
        flask_process.terminate()
