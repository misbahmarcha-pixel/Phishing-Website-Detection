import os
import subprocess
import sys

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def train_model():
    print("🔄 Model not found. Training model...")
    subprocess.run([sys.executable, "train_model.py"])
    print("✅ Model training completed.\n")

def start_app():
    print("🚀 Starting Flask application...\n")
    subprocess.run([sys.executable, "app.py"])

if __name__ == "__main__":
    print("📁 Starting Phishing Detection System...\n")

    # Step 1: Check model
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        print("✅ Model already exists.\n")

    # Step 2: Start Flask app
    start_app()
    
    