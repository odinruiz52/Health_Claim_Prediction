# 🏥 Health Insurance Claim Prediction

This project predicts whether a health insurance claim will be approved or denied based on key features like patient info, procedure type, and claim details.

It includes:
- A Python-based **Flask web app** for making predictions
- A **command-line tool** for quick testing
- A trained machine learning model (`.pkl`) using real-world-style features

---

## 🚀 What You’ll Learn

- How to build a classification model for insurance claims
- How to use Flask to create a simple prediction app
- How to structure and deploy real-world ML projects

---

## 🧠 Tech Stack

- Python, Pandas, scikit-learn
- Flask (for the web interface)
- Command-line I/O for power users
- Preprocessed CSV data (`insurance_encoded.csv`)

---

## 📁 Project Structure

```bash
health-insurance-claim-prediction/
│
├── src/
│   ├── app.py                   # Flask app
│   ├── interactive_prediction.py # CLI tool
│   ├── number.py                # Utility script
│   └── final_claim_model.pkl    # Trained model
│
├── templates/
│   └── index.html               # Web UI
│
├── data/
│   └── insurance_encoded.csv    # Encoded claim dataset
│
└── README.md                    # You're here
