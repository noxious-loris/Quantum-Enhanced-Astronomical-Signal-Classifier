🔭 Quantum-Enhanced Astronomical Signal Classifier

This project is a hybrid quantum-classical machine learning solution designed to detect and classify radio signals from space, helping astronomers filter noise and identify potential pulsars or unknown astronomical events faster.

Built for quantum hackathons, this project demonstrates how quantum feature mapping and variational classifiers (VQC) can enhance radio signal classification compared to traditional methods.
🚀 Features

    ✅ Quantum Machine Learning (QML): Uses Qiskit's Variational Quantum Classifier (VQC).

    ✅ Hybrid Pipeline: Combines classical preprocessing with quantum-enhanced classification.

    ✅ Dashboard: Streamlit web app for uploading and classifying radio telescope data.

    ✅ Datasets: Compatible with SETI, Breakthrough Listen, and HTRU2 Pulsar datasets.

    ✅ Visualization: Displays signal predictions and allows results download.

🧩 Architecture

📂 Dataset Sources

    Breakthrough Listen Open Data

    Kaggle SETI Dataset

    UCI HTRU2 Pulsar Dataset

⚙️ Installation

1️⃣ Clone Repository

git clone https://github.com/yourusername/quantum-signal-classifier.git
cd quantum-signal-classifier


2️⃣ Create Virtual Environment

python3 -m venv quantum_env
source quantum_env/bin/activate   # Windows: quantum_env\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the App

streamlit run quantum_signal_ui.py



🧠 Usage

    Prepare Dataset:

        Use CSV with columns: freq, amp, snr

        Or adapt code for SETI/Kaggle datasets

    Upload File:

        Open the dashboard → Upload your dataset

    Train & Classify:

        Train quantum model (demo mode)

        Classify signals → Download results

🛠️ Tech Stack

    Quantum: Qiskit, Qiskit Machine Learning

    Classical ML: scikit-learn, pandas, numpy

    Visualization: Streamlit, matplotlib

    Data Processing: librosa, plotly
    
📜 License

MIT License – free to use and modify for educational and research purposes.
