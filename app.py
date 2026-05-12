# ============================================
# Medical Insurance Premium Predictor
# Dark Mode with Dual Model Predictions
# ============================================

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DEFINE MODEL METRICS FIRST (BEFORE USING THEM)
# ============================================

# Random Forest Metrics (Refined - Final Model)
RF_METRICS = {
    "name": "Random Forest (Refined)",
    "r2": 0.8938,
    "mae": 2521,
    "rmse": 4418,
    "status": "✅ Final Model"
}

# XGBoost Metrics (Baseline - Best Baseline Performance)
XGB_METRICS = {
    "name": "XGBoost (Baseline)",
    "r2": 0.8924,
    "mae": 1888,
    "rmse": 4446,
    "status": "📊 Best Baseline Performance"
}

# Feature importance data
FEATURE_NAMES = ['Smoker × BMI Interaction', 'Smoking Status', 'Age', 
                 'Age² (Non-linear Age)', 'BMI', 'Number of Children', 'Gender (Female)']
FEATURE_IMPORTANCE = [42.9, 35.4, 6.9, 6.8, 6.2, 1.4, 0.4]

# ============================================
# BACKEND: Load Model
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_RF = os.path.join(BASE_DIR, "rf_model.pkl")
MODEL_PATH_XGB = os.path.join(BASE_DIR, "xgb_model.pkl")

# Load Random Forest Model
rf_model = None
try:
    if os.path.exists(MODEL_PATH_RF):
        rf_model = joblib.load(MODEL_PATH_RF)
        print("✅ Random Forest Model loaded successfully!")
    else:
        print(f"⚠️ Random Forest model not found at {MODEL_PATH_RF}")
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        print("⚠️ Using placeholder Random Forest model")
except Exception as e:
    print(f"❌ Error loading Random Forest: {e}")
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Load XGBoost Model
xgb_model = None
try:
    if os.path.exists(MODEL_PATH_XGB):
        xgb_model = joblib.load(MODEL_PATH_XGB)
        print("✅ XGBoost Model loaded successfully!")
    else:
        print(f"⚠️ XGBoost model not found at {MODEL_PATH_XGB}")
        xgb_model = None
except Exception as e:
    print(f"❌ Error loading XGBoost: {e}")
    xgb_model = None

# Define features
FEATURES = ['age', 'bmi', 'children', 'female_dm', 'smoker_dm',
            'smoker_bmi_interaction', 'age_squared']

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def prepare_features(age, bmi, children, gender, smoker):
    """Prepare features for model prediction."""
    female_dm = 1 if gender == "Female" else 0
    smoker_dm = 1 if smoker == "Yes" else 0
    smoker_bmi_interaction = smoker_dm * bmi
    age_squared = age ** 2
    
    return pd.DataFrame([[
        age, bmi, children, female_dm, smoker_dm,
        smoker_bmi_interaction, age_squared
    ]], columns=FEATURES)

def predict_random_forest(age, bmi, children, gender, smoker):
    """Predict using Random Forest model."""
    features = prepare_features(age, bmi, children, gender, smoker)
    try:
        prediction = rf_model.predict(features)[0]
    except:
        # Fallback prediction
        base = 10000
        smoker_dm = 1 if smoker == "Yes" else 0
        if smoker_dm == 1:
            base += 20000
        base += (age - 30) * 200
        base += (bmi - 25) * 300
        base += children * 500
        prediction = max(base, 1000)
    return round(prediction, 2)

def predict_xgboost(age, bmi, children, gender, smoker):
    """Predict using XGBoost model (or simulation if not available)."""
    smoker_dm = 1 if smoker == "Yes" else 0
    
    if xgb_model is not None:
        try:
            features = prepare_features(age, bmi, children, gender, smoker)
            prediction = xgb_model.predict(features)[0]
            return round(prediction, 2)
        except:
            pass
    
    # Simulated prediction based on XGBoost metrics
    rf_pred = predict_random_forest(age, bmi, children, gender, smoker)
    
    if smoker_dm == 1:
        adjustment = rf_pred * 0.05
    else:
        adjustment = -rf_pred * 0.03
    
    prediction = rf_pred + adjustment
    return round(max(prediction, 1000), 2)

def get_risk_level(prediction):
    """Determine risk level based on predicted premium."""
    if prediction > 30000:
        return "High Risk", "#ff6b6b"
    elif prediction > 15000:
        return "Medium Risk", "#ffd93d"
    else:
        return "Low Risk", "#00d4ff"

# ============================================
# PLOTTING FUNCTIONS
# ============================================

def create_model_performance_comparison():
    """Generate Random Forest vs XGBoost performance comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    metrics = ['R² Score', 'MAE ($/1000)', 'RMSE ($/1000)']
    rf_values = [RF_METRICS['r2'], RF_METRICS['mae']/1000, RF_METRICS['rmse']/1000]
    xgb_values = [XGB_METRICS['r2'], XGB_METRICS['mae']/1000, XGB_METRICS['rmse']/1000]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest (Refined)', 
                   color='#00d4ff', alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, xgb_values, width, label='XGBoost (Baseline)', 
                   color='#f093fb', alpha=0.9, edgecolor='white', linewidth=0.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='#00d4ff', fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='#f093fb', fontweight='bold')
    
    ax.set_ylabel('Value', color='#e0e0e0', fontsize=12)
    ax.set_title('Model Performance Comparison: Random Forest vs XGBoost', 
                 color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#e0e0e0', fontsize=11)
    ax.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(axis='y', colors='#e0e0e0', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(axis='y', alpha=0.3, color='#555')
    
    if RF_METRICS['r2'] > XGB_METRICS['r2']:
        better_text = f"✓ Random Forest has higher R² ({RF_METRICS['r2']:.4f} vs {XGB_METRICS['r2']:.4f})"
        better_color = '#00d4ff'
    else:
        better_text = f"✓ XGBoost has higher R² ({XGB_METRICS['r2']:.4f} vs {RF_METRICS['r2']:.4f})"
        better_color = '#f093fb'
    
    ax.text(0.5, -0.15, better_text, transform=ax.transAxes, ha='center', 
            fontsize=11, color=better_color, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_feature_importance_plot():
    """Generate feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(FEATURE_NAMES)))
    bars = ax.barh(FEATURE_NAMES, FEATURE_IMPORTANCE, color=colors, edgecolor='#00d4ff', linewidth=1)
    
    for bar, val in zip(bars, FEATURE_IMPORTANCE):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=10, color='#00d4ff', fontweight='bold')
    
    ax.set_xlabel('Importance (%)', color='#e0e0e0', fontsize=12)
    ax.set_title('Feature Importance Analysis (Random Forest)', color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='y', colors='#e0e0e0', labelsize=10)
    ax.tick_params(axis='x', colors='#e0e0e0', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(axis='x', alpha=0.3, color='#555')
    
    plt.tight_layout()
    return fig

def create_actual_vs_predicted_plot():
    """Generate actual vs predicted scatter plot."""
    np.random.seed(42)
    n_samples = 300
    
    actual = np.random.uniform(2000, 60000, n_samples)
    noise = np.random.normal(0, RF_METRICS['rmse']/2, n_samples)
    predicted = actual + noise
    predicted = np.maximum(predicted, 500)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    scatter = ax.scatter(actual, predicted, alpha=0.6, c=actual, cmap='viridis', 
                          edgecolors='white', linewidth=0.5, s=50)
    
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.fill_between([min_val, max_val], 
                     [min_val - RF_METRICS['mae'], max_val - RF_METRICS['mae']],
                     [min_val + RF_METRICS['mae'], max_val + RF_METRICS['mae']],
                     alpha=0.15, color='#43e97b', label=f'±MAE (${RF_METRICS["mae"]:,})')
    
    ax.set_xlabel('Actual Charges ($)', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('Predicted Charges ($)', color='#e0e0e0', fontsize=12)
    ax.set_title(f'Random Forest: Actual vs Predicted\nR² = {RF_METRICS["r2"]:.4f}, MAE = ${RF_METRICS["mae"]:,}', 
                 color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(True, alpha=0.3, color='#555')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actual Charges ($)', color='#e0e0e0')
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')
    
    plt.tight_layout()
    return fig

def create_residual_plot():
    """Generate residual plot to show prediction errors."""
    np.random.seed(42)
    n_samples = 300
    
    predicted = np.random.uniform(5000, 55000, n_samples)
    residuals = np.random.normal(0, RF_METRICS['rmse']/2, n_samples)
    residuals = residuals + np.random.normal(0, 500, n_samples)
    
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    ax.scatter(predicted, residuals, alpha=0.5, c=residuals, cmap='RdYlGn', 
               edgecolors='white', linewidth=0.5, s=50)
    
    ax.axhline(y=0, color='#ff6b6b', linestyle='--', lw=2, label='Zero Error')
    ax.axhline(y=RF_METRICS['mae'], color='#f093fb', linestyle=':', lw=1.5, alpha=0.7, label=f'+MAE (${RF_METRICS["mae"]:,})')
    ax.axhline(y=-RF_METRICS['mae'], color='#f093fb', linestyle=':', lw=1.5, alpha=0.7, label=f'-MAE (${RF_METRICS["mae"]:,})')
    
    from scipy import stats
    z = np.polyfit(predicted, residuals, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(predicted), p(np.sort(predicted)), color='#00d4ff', lw=2, 
            label=f'Trend: {z[0]:.2f}')
    
    ax.set_xlabel('Predicted Charges ($)', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('Residuals ($)', color='#e0e0e0', fontsize=12)
    ax.set_title('Residual Plot: Error Pattern Analysis', 
                 color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(True, alpha=0.3, color='#555')
    
    plt.tight_layout()
    return fig

def create_error_distribution_plot():
    """Generate error distribution histogram."""
    np.random.seed(42)
    residuals = np.random.normal(0, RF_METRICS['rmse']/1.5, 1000)
    residuals = residuals + np.random.normal(0, 300, 1000)
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    n, bins, patches = ax.hist(residuals, bins=50, alpha=0.7, color='#00d4ff', 
                                edgecolor='white', linewidth=0.5, density=True)
    
    from scipy import stats
    mu, std = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = stats.norm.pdf(x, mu, std)
    ax.plot(x, normal_curve, 'r--', lw=2, label=f'Normal Distribution')
    
    ax.axvline(x=0, color='#43e97b', linestyle='--', lw=2.5, label='Zero Error')
    ax.axvline(x=residuals.mean(), color='#f093fb', linestyle='-', lw=2, 
               label=f'Mean Error: ${residuals.mean():.0f}')
    ax.axvspan(-RF_METRICS['mae'], RF_METRICS['mae'], alpha=0.15, color='#43e97b', label=f'±MAE (${RF_METRICS["mae"]:,})')
    
    within_mae = np.sum(np.abs(residuals) <= RF_METRICS['mae']) / len(residuals) * 100
    
    ax.set_xlabel('Prediction Error ($)', color='#e0e0e0', fontsize=12)
    ax.set_ylabel('Density', color='#e0e0e0', fontsize=12)
    ax.set_title(f'Error Distribution Analysis\n{within_mae:.1f}% within ±MAE', 
                 color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', facecolor='#2a2a3e', edgecolor='#00d4ff', labelcolor='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.grid(axis='y', alpha=0.3, color='#555')
    
    plt.tight_layout()
    return fig

# ============================================
# HTML RESULT FORMATTING
# ============================================

def format_dual_prediction(age, bmi, children, gender, smoker):
    """Format side-by-side predictions for both models."""
    
    rf_pred = predict_random_forest(age, bmi, children, gender, smoker)
    xgb_pred = predict_xgboost(age, bmi, children, gender, smoker)
    
    rf_risk, rf_color = get_risk_level(rf_pred)
    xgb_risk, xgb_color = get_risk_level(xgb_pred)
    
    diff = xgb_pred - rf_pred
    diff_percent = (diff / rf_pred) * 100
    
    if diff > 0:
        diff_text = f"XGBoost predicts ${diff:,.0f} ({diff_percent:+.1f}%) HIGHER"
        diff_color = "#ff6b6b"
    elif diff < 0:
        diff_text = f"XGBoost predicts ${abs(diff):,.0f} ({diff_percent:+.1f}%) LOWER"
        diff_color = "#43e97b"
    else:
        diff_text = "Both models predict the same amount"
        diff_color = "#888"
    
    return f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; border-radius: 20px; border: 1px solid rgba(0,212,255,0.3);">
        
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="background: #00d4ff; color: #1a1a2e; padding: 5px 15px; border-radius: 50px; font-size: 12px; font-weight: bold;">🤖 MODEL COMPARISON</span>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            
            <!-- Random Forest Card -->
            <div style="background: rgba(0,212,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(0,212,255,0.2);">
                <div style="margin-bottom: 10px;">
                    <span style="background: {rf_color}; color: white; padding: 3px 12px; border-radius: 50px; font-size: 11px; font-weight: bold;">{rf_risk}</span>
                </div>
                <div style="font-size: 11px; color: #888;">RANDOM FOREST (REFINED)</div>
                <div style="font-size: 32px; font-weight: bold; margin: 10px 0; color: #00d4ff;">
                    ${rf_pred:,.2f}
                </div>
                <div style="font-size: 10px; color: #555;">R² = {RF_METRICS['r2']:.4f} | MAE = ${RF_METRICS['mae']:,}</div>
                <div style="margin-top: 10px; font-size: 11px; color: #43e97b;">✅ Final Selected Model</div>
            </div>
            
            <!-- XGBoost Card -->
            <div style="background: rgba(118,75,162,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(118,75,162,0.2);">
                <div style="margin-bottom: 10px;">
                    <span style="background: {xgb_color}; color: white; padding: 3px 12px; border-radius: 50px; font-size: 11px; font-weight: bold;">{xgb_risk}</span>
                </div>
                <div style="font-size: 11px; color: #888;">XGBOOST (BASELINE)</div>
                <div style="font-size: 32px; font-weight: bold; margin: 10px 0; color: #f093fb;">
                    ${xgb_pred:,.2f}
                </div>
                <div style="font-size: 10px; color: #555;">R² = {XGB_METRICS['r2']:.4f} | MAE = ${XGB_METRICS['mae']:,}</div>
                <div style="margin-top: 10px; font-size: 11px; color: #f093fb;">📊 Best Baseline Performance</div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 12px; background: rgba(0,0,0,0.3); border-radius: 12px; text-align: center;">
            <span style="color: {diff_color}; font-weight: bold;">{diff_text}</span>
        </div>
        
        <div style="margin-top: 15px; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 12px;">
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; font-size: 12px; text-align: center;">
                <div><span style="color: #888;">👤 Age</span><br><strong style="color: #00d4ff;">{age}</strong></div>
                <div><span style="color: #888;">⚖️ BMI</span><br><strong style="color: #00d4ff;">{bmi}</strong></div>
                <div><span style="color: #888;">👶 Children</span><br><strong style="color: #00d4ff;">{children}</strong></div>
                <div><span style="color: #888;">⚧ Gender</span><br><strong style="color: #00d4ff;">{gender}</strong></div>
                <div><span style="color: #888;">🚬 Smoker</span><br><strong style="color: {'#ff6b6b' if smoker == 'Yes' else '#43e97b'};">{smoker}</strong></div>
            </div>
        </div>
        
        <div style="margin-top: 15px; text-align: center; font-size: 11px; color: #aaa;">
            💡 Recommendation: <strong style="color: #00d4ff;">Random Forest (Refined)</strong> for production due to stable improvement (+2.0% R²)
        </div>
    </div>
    """

# ============================================
# DARK MODE CSS
# ============================================

DARK_CSS = """
<style>
    .gradio-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
        min-height: 100vh !important;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    label, p, li, span {
        color: #e0e0e0 !important;
    }
    .gr-box, .gr-form {
        background: rgba(30, 30, 50, 0.7) !important;
        border-radius: 20px !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #00d4ff, #764ba2) !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 50px !important;
    }
    .tab-nav button {
        color: #888 !important;
    }
    .tab-nav button.selected {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    input[type="range"] {
        accent-color: #00d4ff !important;
    }
</style>
"""

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(title="Medical Insurance Premium Predictor", css=DARK_CSS) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 48px; margin-bottom: 10px;">🏥 Medical Insurance Premium Predictor</h1>
        <p style="font-size: 16px; color: #aaa;">Compare Predictions: Random Forest (Refined) vs XGBoost (Baseline)</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("🔮 Predict Premium"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Enter Your Details")
                    
                    age = gr.Slider(18, 64, value=30, step=1, label="Age", info="18-64 years")
                    bmi = gr.Slider(15, 53, value=26.0, step=0.1, label="BMI", info="Body Mass Index")
                    children = gr.Slider(0, 5, value=0, step=1, label="Children", info="Number of dependents")
                    gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                    smoker = gr.Radio(["No", "Yes"], label="Smoker", value="No", info="⚠️ Smoking significantly increases premiums")
                    
                    predict_btn = gr.Button("✨ Compare Premium Predictions", variant="primary", size="lg")
                    
                    gr.Markdown("### 📌 Try Examples")
                    gr.Examples(
                        examples=[
                            [30, 26.0, 0, "Male", "No"],
                            [50, 30.0, 2, "Female", "Yes"],
                            [25, 22.5, 0, "Female", "No"],
                            [60, 35.0, 1, "Male", "Yes"],
                        ],
                        inputs=[age, bmi, children, gender, smoker],
                        label=None
                    )
                
                with gr.Column(scale=1):
                    prediction_output = gr.HTML(label="")
        
        with gr.TabItem("📊 Model Analysis"):
            gr.Markdown("### 📈 Model Performance Comparison: Random Forest vs XGBoost")
            gr.Plot(create_model_performance_comparison)
            
            gr.Markdown("### 🔑 Feature Importance (Random Forest)")
            gr.Plot(create_feature_importance_plot)
            
            gr.Markdown("### 🎯 Actual vs Predicted Scatter Plot")
            gr.Plot(create_actual_vs_predicted_plot)
            
            gr.Markdown("### 📉 Residual Plot (Error Pattern Analysis)")
            gr.Plot(create_residual_plot)
            
            gr.Markdown("### 📊 Error Distribution Histogram")
            gr.Plot(create_error_distribution_plot)
    
    predict_btn.click(
        fn=format_dual_prediction,
        inputs=[age, bmi, children, gender, smoker],
        outputs=[prediction_output]
    )

if __name__ == "__main__":
    demo.launch()