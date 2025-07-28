🫀 CardioSense AI – ECG Classification Using Machine Learning

CardioSense AI is a machine learning-based biomedical project that analyzes ECG (Electrocardiogram) signal data to classify heart conditions into categories like **Normal**, **Arrhythmia**, and **Atrial Fibrillation (AFib)**. This project demonstrates how AI can be applied in **healthcare signal processing** using Python.


## 📁 Project Structure

ECG\_AI\_PROJECT/
│
├── data/
│   └── ecg\_phase2\_balanced.csv       # Preprocessed and balanced dataset
│
├── main.py                           # Main script for training and prediction
├── ecg\_plot.py                       # (Optional) Script for signal plotting
├── requirements.txt                  # Required libraries
├── README.md                         # This file

## 📊 Dataset

The dataset used is a CSV file containing extracted features from ECG signals:

| Feature         | Description                                  |
|----------------|----------------------------------------------|
| `avg_hr`        | Average heart rate                          |
| `rr_std`        | Standard deviation of RR intervals          |
| `qrs_count`     | QRS complex count in the signal             |
| `signal_mean`   | Mean value of ECG signal                    |
| `signal_std`    | Standard deviation of ECG signal            |
| `label`         | Class label (Normal, Arrhythmia, AFib)      |


## 🚀 How It Works

1. **Load the ECG dataset** (`ecg_phase2_balanced.csv`)
2. **Preprocess** the data (label encoding, scaling)
3. **Train/test split**
4. **Train a machine learning model** (e.g., RandomForestClassifier)
5. **Evaluate performance** using precision, recall, and accuracy
6. (Optional) **Visualize predictions**


## 🛠️ Tech Stack

- 🐍 Python 3.x
- 📦 Pandas, NumPy
- 📊 Matplotlib, Seaborn
- 🤖 Scikit-learn
- 📁 CSV file as input

## ✅ Example Output

Model Accuracy: 86.67%
Classification Report:
precision    recall  f1-score   support

    AFib       1.00      1.00      1.00         3
Arrhythmia       0.83      0.83      0.83         6
Normal       0.80      0.80      0.80         6
## 🧠 Key Learnings

- Understanding ECG signal components (PQRST, HR, RR interval)
- Feature extraction for biomedical signals
- Applying ML classifiers in healthcare
- Evaluating model performance with real-world noisy data

## 🔗 Correlation with Electronics & Communication Engineering

This project applies **signal processing** concepts — such as filtering, feature extraction, and waveform analysis — from ECE. It bridges **biomedical engineering and AI**, showing how **digital signal processing** (DSP) can be applied to medical diagnostics.


## 📦 Installation
git clone https://github.com/TharaniKumar2004/CardioSenseAI.git
cd CardioSenseAI
pip install -r requirements.txt
python main.py
````

---

## 🧾 License

This project is for educational and research purposes only.

---

## 👨‍💻 Author

**Tharani Kumar**
Final Year ECE Student
Passionate about AI in Healthcare and Signal Processing
