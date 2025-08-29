# 🏥 Health Insurance Claim Prediction System

A comprehensive machine learning system for predicting health insurance claim approval status, built with production-grade security, ethical AI considerations, and healthcare compliance in mind.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Development](#model-development)
- [Ethical AI & Compliance](#ethical-ai--compliance)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system predicts whether health insurance claims will be approved or denied using machine learning algorithms. It's designed for healthcare organizations, insurance companies, and researchers who need reliable, fair, and transparent claim prediction capabilities.

### 🎯 Key Objectives

- **Accuracy**: Achieve high prediction accuracy while maintaining fairness
- **Transparency**: Provide explainable AI decisions for healthcare stakeholders
- **Compliance**: Meet HIPAA, ACA, and other healthcare regulatory requirements
- **Ethics**: Implement bias monitoring and fairness assessments
- **Scalability**: Production-ready architecture with proper error handling

---

## ✨ Features

### 🤖 Machine Learning Capabilities
- **Multiple Algorithm Support**: Logistic Regression, Random Forest, XGBoost, SVM, and more
- **Automated Model Selection**: Cross-validation and hyperparameter tuning
- **Feature Engineering**: Advanced feature creation and selection
- **Performance Monitoring**: Comprehensive model evaluation metrics

### 🌐 Web Interface
- **Modern UI**: Professional, responsive web interface
- **Real-time Predictions**: Instant claim status predictions with confidence scores
- **Input Validation**: Comprehensive data validation and error handling
- **Mobile Friendly**: Responsive design for all devices

### 🔒 Security & Compliance
- **Input Sanitization**: Protection against malicious inputs
- **Error Handling**: Secure error messages without information leakage
- **Audit Logging**: Comprehensive logging for compliance
- **Configuration Management**: Environment-based secure configuration

### ⚖️ Ethical AI
- **Bias Detection**: Automated fairness assessment across demographic groups
- **Regulatory Compliance**: HIPAA, ACA compliance checking
- **Model Explainability**: SHAP values and feature importance analysis
- **Patient Rights**: Documentation and implementation of patient rights

### 📊 Analytics & Reporting
- **Exploratory Data Analysis**: Comprehensive EDA notebook
- **Performance Visualizations**: Charts and plots for model insights
- **Ethical AI Reports**: Automated bias and fairness reporting
- **Audit Trails**: Complete decision tracking and documentation

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/health-insurance-claim-prediction.git
cd health-insurance-claim-prediction

# Install dependencies
pip install -r requirements.txt

# Run the web application
cd src
python app.py

# Or use the command-line interface
python interactive_prediction.py
```

Visit `http://localhost:5000` to access the web interface.

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Step-by-Step Installation

1. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import pandas, sklearn, flask; print('All dependencies installed successfully!')"
   ```

4. **Run Initial Setup**
   ```bash
   python config.py  # Verify configuration
   python model_development.py  # Train models (optional)
   ```

---

## 📁 Project Structure

```
health-insurance-claim-prediction/
│
├── 📊 Data & Analysis
│   ├── data/
│   │   └── insurance_encoded.csv           # Dataset
│   ├── exploratory_data_analysis.ipynb    # EDA notebook
│   └── model_development.py               # ML pipeline
│
├── 🌐 Web Application
│   ├── src/
│   │   ├── app.py                         # Flask web server
│   │   ├── interactive_prediction.py     # CLI interface
│   │   └── templates/
│   │       └── index.html                 # Web UI
│
├── ⚙️ Configuration & Core
│   ├── config.py                          # Configuration management
│   ├── ethical_ai_framework.py           # Ethical AI toolkit
│   └── requirements.txt                   # Dependencies
│
├── 📈 Models & Results
│   ├── models/                            # Trained models
│   ├── results/                           # Performance metrics
│   ├── plots/                             # Visualizations
│   └── ethical_ai_reports/                # Fairness reports
│
├── 📚 Documentation
│   ├── README.md                          # This file
│   └── docs/                              # Additional documentation
│
└── 🧪 Testing & Quality
    └── tests/                             # Unit tests
```

---

## 📖 Usage Guide

### 🌐 Web Interface

1. **Start the Web Server**
   ```bash
   cd src
   python app.py
   ```

2. **Access the Interface**
   - Open browser to `http://localhost:5000`
   - Fill in patient/claim information
   - Click "Predict Claim Status"
   - View prediction with confidence score

3. **Input Fields**
   - **Age**: Patient age (18-100)
   - **Sex**: 0=Female, 1=Male
   - **BMI**: Body Mass Index (15.0-50.0)
   - **Children**: Number of dependents (0-10)
   - **Smoker**: 0=No, 1=Yes
   - **Region**: 0-3 (geographical regions)
   - **Charges**: Medical charges ($0-$100,000)

### 💻 Command Line Interface

1. **Start CLI Tool**
   ```bash
   cd src
   python interactive_prediction.py
   ```

2. **Available Options**
   - Option 1: Predict using existing dataset record
   - Option 2: Input new patient data
   - Option 3: View dataset statistics
   - Option 4: Exit

### 🔧 Configuration

Environment variables can be set to customize behavior:

```bash
# Model configuration
export MODEL_PATH="models/custom_model.pkl"
export PREDICTION_THRESHOLD="0.6"

# Flask configuration
export FLASK_DEBUG="True"
export FLASK_PORT="8080"

# Logging configuration
export LOG_LEVEL="DEBUG"
export LOG_FILE="logs/custom.log"
```

---

## 🧠 Model Development

### Training New Models

```bash
# Run complete ML pipeline
python model_development.py
```

This script will:
1. Load and preprocess data
2. Perform feature engineering
3. Train multiple algorithms
4. Evaluate model performance
5. Select best model
6. Save trained model and metrics

### Model Comparison

The pipeline automatically compares:
- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting for high performance
- **SVM**: Support Vector Machine for complex boundaries
- **K-Nearest Neighbors**: Instance-based learning

### Performance Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate (important for claim approval)
- **Recall**: Sensitivity (important for catching denied claims)
- **F1-Score**: Balanced metric
- **ROC-AUC**: Area under the curve

### Feature Engineering

Automatic feature creation includes:
- Age groups (Young, Middle, Senior, Elder)
- BMI categories (Underweight, Normal, Overweight, Obese)
- High-cost claim indicators
- Interaction terms (smoker × age, smoker × BMI)

---

## ⚖️ Ethical AI & Compliance

### Fairness Assessment

```bash
# Generate ethical AI report
from ethical_ai_framework import EthicalAIFramework
ethical_ai = EthicalAIFramework(model, data, predictions)
report = ethical_ai.generate_comprehensive_report()
```

### Bias Monitoring

The system automatically monitors for bias across:
- **Gender**: Male vs Female approval rates
- **Age Groups**: Different age cohort fairness
- **Geographic Regions**: Regional disparities
- **Other Demographics**: Extensible to additional protected attributes

### Regulatory Compliance

#### HIPAA Compliance
- ✅ PHI protection through data anonymization
- ✅ Minimum necessary data usage
- ⚠️ Access controls (implement authentication)
- ✅ Comprehensive audit trails

#### ACA Compliance
- ⚠️ Non-discrimination monitoring required
- ✅ No explicit pre-existing condition bias
- ✅ Transparent decision-making process

### Patient Rights

The system supports:
- **Right to Explanation**: Feature importance and SHAP explanations
- **Right to Human Review**: Appeal process for disputed decisions
- **Right to Data Access**: View data used in predictions
- **Right to Correction**: Process to correct inaccurate information

---

## 🔌 API Documentation

### Prediction Endpoint

```http
POST /predict
Content-Type: application/x-www-form-urlencoded

age=35&sex=1&bmi=25.5&children=2&smoker=0&region=1&charges=5000.00
```

**Response:**
```json
{
  "prediction": "Approved",
  "confidence": 0.85,
  "status": "success"
}
```

### Health Check Endpoint

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Test Coverage
```bash
pytest --cov=src tests/
```

### Manual Testing
```bash
# Test web interface
curl -X POST http://localhost:5000/predict \
  -d "age=30&sex=0&bmi=22.5&children=1&smoker=0&region=2&charges=3000"

# Test CLI interface
echo "2\n30\n0\n22.5\n1\n0\n2\n3000" | python interactive_prediction.py
```

---

## 📈 Performance Monitoring

### Model Metrics Dashboard

The system tracks:
- **Prediction Accuracy**: Real-time accuracy monitoring
- **Response Time**: API response performance
- **Error Rates**: System reliability metrics
- **Bias Metrics**: Ongoing fairness monitoring

### Alerting

Configure alerts for:
- Model performance degradation
- Bias threshold violations
- System errors or downtime
- Compliance violations

---

## 🔧 Troubleshooting

### Common Issues

**Model Not Found Error**
```bash
# Solution: Check model path or retrain model
export MODEL_PATH="models/final_claim_model.pkl"
python model_development.py
```

**Port Already in Use**
```bash
# Solution: Use different port
export FLASK_PORT="8080"
python app.py
```

**Memory Issues**
```bash
# Solution: Reduce model complexity or increase system memory
# Edit model_development.py to use smaller models
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL="DEBUG"
export FLASK_DEBUG="True"
python app.py
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for changes
- Ensure ethical AI compliance

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/health-insurance-claim-prediction/issues)
- **Documentation**: [Project Wiki](https://github.com/your-username/health-insurance-claim-prediction/wiki)
- **Email**: Contact via GitHub issues

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Healthcare data science community
- Open source ML libraries (scikit-learn, pandas, flask)
- Ethical AI researchers and practitioners
- Healthcare compliance experts

---

**⚡ Built with ❤️ for better healthcare outcomes**

*Last updated: $(date +'%Y-%m-%d')*
