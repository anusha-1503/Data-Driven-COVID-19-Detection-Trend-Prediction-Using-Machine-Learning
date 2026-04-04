# Data-Driven-COVID-19-Detection-Trend-Prediction-Using-Machine-Learning
ME 228 Course Project | Domain: Biomedical Imaging + Epidemiological Time-Series Forecasting |
## About the Project
Managing a global pandemic requires both rapid clinical tools and accurate predictive models. This project addresses two critical challenges:
* **Clinical Detection:** Standard PCR testing is resource-intensive. We provide a multi-class X-ray classifier to distinguish COVID-19 from other pulmonary conditions with high precision.
* **Epidemiological Forecasting:** Predicting future case trajectories is vital for public health decisions. Our model captures non-stationary effects like lockdowns and viral variants that classical models miss.

## Key Features
* **Explainable AI (XAI):** Uses Grad-CAM heatmaps to highlight radiological markers (like opacities) for clinical transparency.
* **Leakage Prevention:** Implements Patient-Level Splitting (GroupShuffleSplit) to ensure a model does not "memorize" specific patient anatomy.
* **Exogenous Integration:** Incorporates Google Mobility and Government Stringency Indices to improve forecasting accuracy.
* **Hybrid Deep Learning:** Combines Transformers for long-range dependencies with LSTMs for local sequential trends.

## Dataset Overview

| Module | Dataset | Size/Scope |
| :--- | :--- | :--- |
| **Vision** | [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | 21,165 labeled CXR images |
| **Forecasting** | OWID / JHU CSSE | Daily cases/deaths across 200+ countries |
| **Exogenous** | Google Health Open Data | Mobility, weather, and demographics |

## Model Architecture
### Module 1: Vision (EfficientNet-B4)
* Preprocessing: CLAHE normalization and data augmentation.
* Strategy: Two-stage transfer learning (frozen backbone for 5 epochs before full fine-tuning).
* Verification: Grad-CAM integration for radiological marker verification.

### Module 2: Forecasting (Transformer-LSTM Hybrid)
* **Architecture:** A fused model where the Transformer encoder processes global trends and the LSTM captures daily fluctuations.
* **Output:** Generates 7, 14, and 30-day multi-horizon probabilistic forecasts.
* **Optimization:** Automated hyperparameter tuning via Optuna.
