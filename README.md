# 🏥 Medical Insurance Premium Predictor

## 📋 Overview

This application predicts annual medical insurance costs based on personal health information. It compares two machine learning models:

- **Random Forest (Refined)** – Final selected model for production
- **XGBoost (Baseline)** – AI-recommended model with strong baseline performance

The models were developed as part of a university examination on predictive analytics, using real medical insurance data.

## 📊 Model Performance

| Metric | Random Forest (Refined) | XGBoost (Baseline) |
|--------|------------------------|-------------------|
| **R² Score** | 0.8938 | 0.8924 |
| **MAE** | $2,521 | $1,888 |
| **RMSE** | $4,418 | $4,446 |
| **Improvement** | ✅ +2.0% from baseline | ❌ -3.8% after refinement |

**Final Decision:** Random Forest (Refined) selected for production due to stable improvement and lower RMSE.

## 🔑 Key Insights

| Feature | Importance |
|---------|------------|
| Smoker × BMI Interaction | 42.9% |
| Smoking Status | 35.4% |
| Age | 6.9% |
| Age² (Non-linear Age) | 6.8% |
| BMI | 6.2% |

### Business Impact

- **Smoking-BMI interaction** is the strongest predictor → target this segment for premium adjustments
- **Smokers pay ~$23,610 more** on average → implement smoker verification
- **Gender has minimal impact** (0.4%) → avoid gender-based pricing
- **Annual savings:** $1.94M per 10,000 policies from refinement

## 📁 Dataset

- **Source:** Medical Cost Personal Dataset (Kaggle)
- **Records:** 1,337 patients
- **Features:** Age, BMI, children, gender, smoking status
- **Target:** Annual medical charges ($1,122 – $63,770)

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, scikit-learn, XGBoost |
| Frontend | Gradio |
| Deployment | Hugging Face Spaces |
| Visualization | Matplotlib |

## 🚀 How to Use

1. Enter your personal details (age, BMI, children, gender, smoking status)
2. Click "Compare Premium Predictions"
3. View side-by-side predictions from both models
4. Explore the "Model Analysis" tab for visualizations

## 📈 Visualizations Included

- Model Performance Comparison (Baseline vs Refined)
- Feature Importance Bar Chart
- Actual vs Predicted Scatter Plot
- Residual Plot (Error Pattern Analysis)
- Error Distribution Histogram

## 👤 Author

**Nathanphil-Z9 Gordo Bacay** – University of Mindanao

## 📝 License

MIT

## 🔗 Links

- [Live Demo](https://enneanine-insurance-predictor.hf.space)
- [Dataset Source](https://www.kaggle.com/datasets/mirichoi0218/insurance)
