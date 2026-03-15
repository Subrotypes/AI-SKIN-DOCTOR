AI-SKIN-DOCTOR 🩺
AI-SKIN-DOCTOR is a dual-layered medical assistant that detects skin infections. It utilizes a custom-trained CNN model for offline, real-time classification and integrates the Gemini API for comprehensive, context-aware analysis when online.

🚀 Features
Offline Mode: Uses a MobileNetV2 based model (skin_model.keras) to classify skin images as "Normal" or "Infected."

AI Consultant: If an internet connection is available, the app uses Gemini to provide a detailed clinical description and suggest potential symptoms to watch for.

Privacy First: Users can run the local model without uploading data to the cloud.

Streamlit Interface: A clean, user-friendly web interface for easy image uploads and result visualization.

🛠️ Tech Stack
Frameworks: TensorFlow / Keras, Streamlit

Languages: Python

AI/ML: CNN (Convolutional Neural Networks), Google Gemini API

Environment: python-dotenv for secure API key management

📦 Installation & Setup
Clone the repository:

Bash
git clone https://github.com/Subrotypes/AI-SKIN-DOCTOR.git
cd AI-SKIN-DOCTOR
Install dependencies:

Bash
pip install -r requirements.txt
Set up your Environment Variables:
Create a .env file in the root directory and add your Gemini API key:

Code snippet
GEMINI_API_KEY=your_api_key_here
Run the Application:

Bash
streamlit run app.py
🧠 Model Training
The local model was trained using TensorFlow/Keras with a dataset of infected and normal skin images.

Base Model: MobileNetV2 (Transfer Learning)

Epochs: 18

Output: Binary classification with confidence scoring.

How to push this to GitHub:
Create the file: code README.md (or just create it in your explorer).

Paste the content above.

Run these commands in your terminal:

Bash
git add README.md
git commit -m "Add professional README"
git push
