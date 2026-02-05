# setup.py

#!/usr/bin/env python3
"""
Setup script for Fake News Detector Streamlit App.
Run this once to install dependencies and download NLTK data.
"""

import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_data():
    print("Downloading NLTK stopwords...")
    import nltk
    nltk.download('stopwords', quiet=True)

def check_model_files():
    missing = [f for f in ("model.pkl","vectorizer.pkl") if not os.path.exists(f)]
    if missing:
        print(f"⚠️ Missing files: {', '.join(missing)}")
    else:
        print("✅ All model files are present.")

def main():
    print("🚀 Setting up Fake News Detector App")
    print("="*40)
    install_requirements()
    download_nltk_data()
    check_model_files()
    print("="*40)
    print("Setup complete! Run the app with: streamlit run app.py")

if __name__ == "__main__":
    main()