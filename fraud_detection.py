import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import json
import warnings

# Suppress scikit-learn version warnings for model compatibility
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------- Page config ----------
st.set_page_config(
    page_title="Fraud Analytics · Houda El Barehoumi",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Global CSS (branding + mobile) ----------
st.markdown("""
<style>
/* fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

/* hero - mobile responsive */
.hero {
  padding: 20px 16px;
  border-radius: 12px;
  background: linear-gradient(135deg, #EEF3FF 0%, #F9FBFF 100%);
  border: 1px solid #E6ECFF;
  margin-bottom: 8px;
}
.hero h1 {
  margin: 0;
  font-weight: 800;
  letter-spacing: -0.02em;
  font-size: 1.8rem;
}
.subtle { 
  color: #475569; 
  font-size: 0.9rem;
  margin-top: 4px;
}

/* cards - mobile responsive */
.card {
  background: #FFFFFF;
  border: 1px solid #EDEFF5;
  border-radius: 12px;
  padding: 16px 12px;
  margin-bottom: 12px;
}
.kpi .stMetric { 
  padding: 6px 0; 
  font-size: 0.9rem;
}

/* footer */
.footer {
  position: sticky; 
  bottom: 0; 
  width: 100%;
  padding: 8px 0; 
  margin-top: 20px;
  color: #64748B; 
  font-size: 12px; 
  text-align: center;
}

/* mobile responsive tweaks */
.block-container { 
  padding-top: 0.5rem; 
  padding-left: 1rem;
  padding-right: 1rem;
}

/* tabs - mobile responsive */
.stTabs [data-baseweb="tab-list"] { 
  gap: 4px; 
  flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
  height: 40px; 
  padding-top: 6px; 
  padding-bottom: 6px;
  padding-left: 12px;
  padding-right: 12px;
  border-radius: 8px 8px 0 0; 
  background: #F3F6FF;
  border: 1px solid #E6ECFF;
  font-size: 0.9rem;
}

/* sliders - mobile friendly */
.stSlider > div > div > div > div { 
  background: #4C7AF2; 
}

/* buttons - mobile responsive */
.stButton > button {
  border-radius: 8px; 
  padding: 0.5rem 0.8rem; 
  font-weight: 600;
  border: 1px solid #e5e7eb;
  font-size: 0.9rem;
  min-height: 44px; /* touch-friendly */
}

/* sidebar - mobile responsive */
.css-1d391kg {
  padding-top: 1rem;
}

/* metrics - mobile responsive */
[data-testid="metric-container"] {
  background: white;
  border: 1px solid #E5E7EB;
  padding: 1rem;
  border-radius: 8px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

[data-testid="metric-container"] > div {
  justify-content: space-between;
}

[data-testid="metric-container"] label {
  font-size: 0.8rem;
  font-weight: 500;
  color: #6B7280;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
  font-size: 1.2rem;
  font-weight: 700;
  color: #111827;
}

/* dataframes - mobile responsive */
.stDataFrame {
  font-size: 0.8rem;
}

/* columns - mobile responsive */
.stColumns > div {
  padding: 0.25rem;
}

/* expanders - mobile responsive */
.streamlit-expanderHeader {
  font-size: 0.9rem;
  padding: 0.75rem 1rem;
}

.streamlit-expanderContent {
  padding: 1rem;
}

/* file uploader - mobile responsive */
.stFileUploader {
  margin-bottom: 1rem;
}

/* number input - mobile responsive */
.stNumberInput > div > div > input {
  min-height: 44px;
  font-size: 1rem;
}

/* selectbox - mobile responsive */
.stSelectbox > div > div > div {
  min-height: 44px;
}

/* multiselect - mobile responsive */
.stMultiSelect > div > div > div {
  min-height: 44px;
}

/* checkbox - mobile responsive */
.stCheckbox > label > div {
  min-height: 44px;
  display: flex;
  align-items: center;
}

/* info/warning/error boxes - mobile responsive */
.stAlert {
  font-size: 0.9rem;
  padding: 0.75rem;
  margin: 0.5rem 0;
}

/* mobile breakpoints */
@media (max-width: 768px) {
  .hero {
    padding: 16px 12px;
    margin: 0 -1rem 1rem -1rem;
    border-radius: 0;
    border-left: none;
    border-right: none;
  }
  
  .hero h1 {
    font-size: 1.5rem;
  }
  
  .block-container {
    padding-left: 0.5rem;
    padding-right: 0.5rem;
  }
  
  .card {
    padding: 12px 8px;
  }
  
  /* stack KPI columns on mobile */
  .stColumns > div[data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
  }
  
  /* make tabs full width on mobile */
  .stTabs [data-baseweb="tab"] {
    flex: 1;
    text-align: center;
  }
  
  /* adjust metric containers for mobile */
  [data-testid="metric-container"] {
    padding: 0.75rem;
    margin-bottom: 0.5rem;
  }
  
  [data-testid="metric-container"] label {
    font-size: 0.75rem;
  }
  
  [data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1rem;
  }
}

@media (max-width: 480px) {
  .hero h1 {
    font-size: 1.3rem;
  }
  
  .subtle {
    font-size: 0.8rem;
  }
  
  .block-container {
    padding-left: 0.25rem;
    padding-right: 0.25rem;
  }
  
  .stTabs [data-baseweb="tab"] {
    font-size: 0.8rem;
    padding-left: 8px;
    padding-right: 8px;
  }
}
</style>
""", unsafe_allow_html=True)

# ---------- Hero header ----------
st.markdown(f"""
<div class="hero">
  <h1>Fraud Detection Analytics</h1>
  <div class="subtle">by <b>Houda El Barehoumi</b> — interactive EDA, risk scoring & explainability</div>
</div>
""", unsafe_allow_html=True)

# ---------- Utils ----------
@st.cache_data(show_spinner=False)
def load_data_df(src):
    import pandas as pd
    df = pd.read_csv(src)
    return df

@st.cache_resource(show_spinner=True)
def load_model_resource():
    import os, joblib
    for p in ["fraud_detection_pipeline.pkt", "fraud_detection_pipeline.pkl"]:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception as e:
                st.error(f"Error loading model {p}: {str(e)}")
                continue
    return None

def get_target_col(df: pd.DataFrame) -> str:
    candidates = ["isFraud", "is_fraud", "fraud", "label"]
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def standard_columns() -> list:
    return ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

def safe_select_X(df):
    cols = standard_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, missing
    return df[cols].copy(), []

def proba(model, X):
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:,1]
        if hasattr(model, "decision_function"):
            import numpy as np
            s = model.decision_function(X)
            return 1/(1+np.exp(-s))
        return None
    except Exception as e:
        error_msg = str(e)
        if any(keyword in error_msg.lower() for keyword in ["_name_to_fitted_passthrough", "isnan", "ufunc", "input types"]):
            # Silent fallback for known compatibility issues
            try:
                if hasattr(model, "predict"):
                    predictions = model.predict(X)
                    return predictions.astype(float)
            except:
                pass
        else:
            st.warning(f"Model prediction error: {error_msg}")
        return None

def kpi_card(col, label, value):
    col.metric(label, value)

def altair_available():
    try:
        import altair as alt  # noqa: F401
        return True
    except Exception:
        return False

def format_number(num):
    return f"{num:,}" if isinstance(num, (int, float)) else str(num)

# Initialize session state
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5

# Load model
model = load_model_resource()

# Model loaded successfully

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Houda El Barehoumi")
    st.caption("Data Science • Fraud Analytics")
    # quick links (edit URLs)
    st.markdown("[LinkedIn](https://www.linkedin.com/in/houda-el-barehoumi-063ab22a2/) · "
                "[GitHub](https://github.com/HUUDAA6) · "
                "[Email](mailto:houdaelbarehoumi@gmail.com)")
    st.markdown("---")
    st.markdown("#### Controls")
    threshold = st.slider("Decision Threshold", 0.01, 0.99, st.session_state.threshold, 0.01, 
                         help="Probability threshold for fraud classification")
    st.session_state.threshold = threshold
    st.markdown("---")

# ---------- Main Tabs ----------
tab1, tab2 = st.tabs(["Explore", "Predict"])

# ---------- Explore ----------
with tab1:
    st.title("Explore Transactions")
    data_source = None
    if os.path.exists("AIML Dataset.csv"):
        data_source = "AIML Dataset.csv"

    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        data_source = uploaded

    if not data_source:
        st.info("Place `AIML Dataset.csv` in the app folder or upload a CSV to explore.")
        st.stop()

    df = load_data_df(data_source)
    if df.empty:
        st.warning("Loaded dataset is empty.")
        st.stop()

    target = get_target_col(df)
    if not target:
        st.warning("Target column not found. Expected one of: isFraud, is_fraud, fraud, label.")
    
    # Enhanced Filters
    st.subheader("Filters")
    
    # Mobile-friendly filter layout
    type_values = df["type"].dropna().unique().tolist() if "type" in df.columns else []
    chosen_types = st.multiselect("Transaction Type", options=type_values, default=type_values, key="explore_transaction_type")
    
    min_amt = float(np.nanmin(df["amount"])) if "amount" in df.columns else 0.0
    max_amt = float(np.nanmax(df["amount"])) if "amount" in df.columns else 0.0
    amt_range = st.slider("Amount range", min_value=float(min_amt), max_value=float(max_amt), 
                         value=(float(min_amt), float(max_amt)), key="explore_amount_range") if "amount" in df.columns else (0.0, 0.0)
    
    # Performance options in expander for mobile
    with st.expander("Performance Options", expanded=False):
        sample_option = st.checkbox("Sample for faster charts", value=len(df) > 200000, key="explore_sample_option")
        sample_size = st.slider("Sample size", 1000, min(200000, len(df)), min(100000, len(df)), 
                               disabled=not sample_option, key="explore_sample_size")
        
        # Date/step range filter
        date_filter = None
        if "step" in df.columns:
            step_range = st.slider("Step range", int(df["step"].min()), int(df["step"].max()), 
                                  (int(df["step"].min()), int(df["step"].max())), key="explore_step_range")
            date_filter = ("step", step_range)
        elif "date" in df.columns:
            try:
                df["date_parsed"] = pd.to_datetime(df["date"])
                date_range = st.date_input("Date range", 
                                         value=(df["date_parsed"].min().date(), df["date_parsed"].max().date()),
                                         min_value=df["date_parsed"].min().date(),
                                         max_value=df["date_parsed"].max().date())
                date_filter = ("date", date_range)
            except:
                pass

    # Apply filters
    fdf = df.copy()
    if "type" in fdf.columns and chosen_types:
        fdf = fdf[fdf["type"].isin(chosen_types)]
    if "amount" in fdf.columns:
        fdf = fdf[(fdf["amount"] >= amt_range[0]) & (fdf["amount"] <= amt_range[1])]
    
    # Apply date/step filter
    if date_filter:
        col_name, range_val = date_filter
        if col_name == "step":
            fdf = fdf[(fdf["step"] >= range_val[0]) & (fdf["step"] <= range_val[1])]
        elif col_name == "date":
            try:
                fdf["date_parsed"] = pd.to_datetime(fdf["date"])
                fdf = fdf[(fdf["date_parsed"].dt.date >= range_val[0]) & 
                         (fdf["date_parsed"].dt.date <= range_val[1])]
            except:
                pass
    
    # Apply sampling if requested
    if sample_option and len(fdf) > sample_size:
        fdf = fdf.sample(n=sample_size, random_state=42)
        st.info(f"Showing {sample_size:,} sampled transactions for faster processing")

    # Enhanced KPIs
    total_tx = len(fdf)
    frauds = int(fdf[target].sum()) if target and fdf.shape[0] else 0
    rate = (frauds / total_tx * 100) if total_tx else 0.0
    
    # Additional KPIs
    total_exposure = float(fdf[fdf[target] == 1]["amount"].sum()) if target and "amount" in fdf.columns else 0.0
    median_fraud_amt = float(fdf[fdf[target] == 1]["amount"].median()) if target and "amount" in fdf.columns else 0.0
    
    st.markdown('<div class="card kpi">', unsafe_allow_html=True)
    
    # Use responsive columns - 2 on mobile, 5 on desktop
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        st.metric("Total Transactions", format_number(total_tx))
    with col2:
        st.metric("Frauds", format_number(frauds))
    with col3:
        st.metric("Fraud Rate", f"{rate:.2f}%")
    with col4:
        st.metric("Total Exposure", f"${total_exposure:,.0f}")
    with col5:
        st.metric("Median Fraud Amount", f"${median_fraud_amt:,.0f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts (Altair preferred; fallback to matplotlib)
    use_altair = altair_available()

    if target:
        st.subheader("Fraud vs Non-Fraud Count")
        if use_altair:
            import altair as alt
            cnt = fdf[target].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
            chart = alt.Chart(cnt).mark_bar().encode(x=alt.X("class:N", title="isFraud"), y="count:Q", tooltip=["class", "count"])
            st.altair_chart(chart, use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            c = fdf[target].value_counts(dropna=False)
            fig = plt.figure()
            c.plot(kind="bar")
            plt.xlabel("isFraud")
            plt.ylabel("count")
            st.pyplot(fig)

    if "type" in fdf.columns and target:
        st.subheader("Fraud rate by Transaction Type")
        grp = fdf.groupby("type")[target].mean().sort_values(ascending=False).reset_index()
        grp[target] = (grp[target] * 100).round(2)
        if altair_available():
            import altair as alt
            chart = alt.Chart(grp).mark_bar().encode(x=alt.X("type:N", sort="-y"), y=alt.Y(f"{target}:Q", title="Fraud rate (%)"), tooltip=["type", target])
            st.altair_chart(chart, use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.bar(grp["type"], grp[target])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Fraud rate (%)")
            st.pyplot(fig)

    if "amount" in fdf.columns and target:
        st.subheader("Amount Distribution (Fraud vs Non-Fraud)")
        left, right = st.columns(2)
        import matplotlib.pyplot as plt
        # Non-fraud
        with left:
            fig1 = plt.figure()
            fdf[fdf[target] == 0]["amount"].plot(kind="hist", bins=30)
            plt.title("Non-Fraud amounts")
            plt.xlabel("amount")
            st.pyplot(fig1)
        # Fraud
        with right:
            fig2 = plt.figure()
            fdf[fdf[target] == 1]["amount"].plot(kind="hist", bins=30)
            plt.title("Fraud amounts")
            plt.xlabel("amount")
            st.pyplot(fig2)

    # Time series
    if "step" in fdf.columns and target:
        st.subheader("Fraud Count by Step (hour)")
        ts = fdf.groupby("step")[target].sum().reset_index()
        if altair_available():
            import altair as alt
            chart = alt.Chart(ts).mark_line().encode(x="step:Q", y=alt.Y(f"{target}:Q", title="fraud count"), tooltip=["step", target])
            st.altair_chart(chart, use_container_width=True)
        else:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(ts["step"], ts[target])
            plt.xlabel("step (hour)")
            plt.ylabel("fraud count")
            st.pyplot(fig)

    # Advanced Analytics Section
    if model is not None and target:
        st.subheader("Advanced Analytics")
        
        # Get prediction features
        X, missing_cols = safe_select_X(fdf)
        
        if X is not None and len(fdf) > 0:
            # Limit sample size for performance
            max_sample = min(100000, len(X))
            if len(X) > max_sample:
                sample_idx = np.random.choice(len(X), max_sample, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = fdf[target].iloc[sample_idx]
            else:
                X_sample = X
                y_sample = fdf[target]
            
            # Get probabilities
            probabilities = proba(model, X_sample)
            
            if probabilities is not None:
                # PR and ROC Curves
                try:
                    from sklearn.metrics import precision_recall_curve, roc_curve, auc
                    
                    # Precision-Recall Curve
                    precision, recall, pr_thresholds = precision_recall_curve(y_sample, probabilities)
                    pr_auc = auc(recall, precision)
                    
                    # ROC Curve
                    fpr, tpr, roc_thresholds = roc_curve(y_sample, probabilities)
                    roc_auc = auc(fpr, tpr)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Precision-Recall Curve")
                        if altair_available():
                            import altair as alt
                            pr_data = pd.DataFrame({'Recall': recall, 'Precision': precision})
                            chart = alt.Chart(pr_data).mark_line().encode(
                                x='Recall:Q', y='Precision:Q'
                            ).properties(title=f'PR-AUC: {pr_auc:.3f}')
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots()
                            ax.plot(recall, precision)
                            ax.set_xlabel('Recall')
                            ax.set_ylabel('Precision')
                            ax.set_title(f'PR-AUC: {pr_auc:.3f}')
                            st.pyplot(fig)
                        st.caption("Higher is better. Shows precision-recall tradeoff.")
                    
                    with col2:
                        st.subheader("ROC Curve")
                        if altair_available():
                            import altair as alt
                            roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
                            chart = alt.Chart(roc_data).mark_line().encode(
                                x='FPR:Q', y='TPR:Q'
                            ).properties(title=f'ROC-AUC: {roc_auc:.3f}')
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots()
                            ax.plot(fpr, tpr)
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title(f'ROC-AUC: {roc_auc:.3f}')
                            st.pyplot(fig)
                        st.caption("Higher is better. Shows true vs false positive tradeoff.")
                
                except ImportError:
                    st.info("Install scikit-learn for PR/ROC curves")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                # Threshold slider for confusion matrix
                matrix_threshold = st.slider("Classification Threshold", 0.01, 0.99, threshold, 0.01, key="confusion_matrix_threshold")
                
                # Calculate confusion matrix
                y_pred = (probabilities >= matrix_threshold).astype(int)
                
                # Calculate metrics
                tp = np.sum((y_sample == 1) & (y_pred == 1))
                fp = np.sum((y_sample == 0) & (y_pred == 1))
                tn = np.sum((y_sample == 0) & (y_pred == 0))
                fn = np.sum((y_sample == 1) & (y_pred == 0))
                
                precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Display confusion matrix
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    cm_data = pd.DataFrame({
                        'Predicted': ['Non-Fraud', 'Fraud', 'Non-Fraud', 'Fraud'],
                        'Actual': ['Non-Fraud', 'Non-Fraud', 'Fraud', 'Fraud'],
                        'Count': [tn, fp, fn, tp],
                        'Percentage': [
                            f"{tn/len(y_sample)*100:.1f}%",
                            f"{fp/len(y_sample)*100:.1f}%", 
                            f"{fn/len(y_sample)*100:.1f}%",
                            f"{tp/len(y_sample)*100:.1f}%"
                        ]
                    })
                    st.dataframe(cm_data, use_container_width=True)
                
                with col2:
                    st.metric("Precision", f"{precision_val:.3f}")
                    st.metric("Recall", f"{recall_val:.3f}")
                    st.metric("F1-Score", f"{f1_val:.3f}")
                    st.metric("Specificity", f"{specificity:.3f}")
                
                # Cost Analysis
                st.subheader("Cost Analysis")
                
                with st.expander("Cost Parameters", expanded=False):
                    cost_fp = st.number_input("False Positive Cost ($)", value=5.0, min_value=0.0, key="cost_fp_input")
                    cost_fn_factor = st.number_input("False Negative Cost Factor", value=1.0, min_value=0.0, 
                                                   help="Multiplied by transaction amount", key="cost_fn_factor_input")
                
                # Calculate expected cost for current threshold
                expected_cost = (fp * cost_fp) + (fn * cost_fn_factor * fdf[fdf[target] == 1]["amount"].mean() if "amount" in fdf.columns else fn * 100)
                
                st.metric("Expected Cost (Current Threshold)", f"${expected_cost:,.2f}")
                
                # Cost vs Threshold curve
                try:
                    thresholds = np.linspace(0.01, 0.99, 20)
                    costs = []
                    
                    for thresh in thresholds:
                        y_pred_thresh = (probabilities >= thresh).astype(int)
                        tp_thresh = np.sum((y_sample == 1) & (y_pred_thresh == 1))
                        fp_thresh = np.sum((y_sample == 0) & (y_pred_thresh == 1))
                        fn_thresh = np.sum((y_sample == 1) & (y_pred_thresh == 0))
                        
                        cost_thresh = (fp_thresh * cost_fp) + (fn_thresh * cost_fn_factor * (fdf[fdf[target] == 1]["amount"].mean() if "amount" in fdf.columns else 100))
                        costs.append(cost_thresh)
                    
                    optimal_idx = np.argmin(costs)
                    optimal_threshold = thresholds[optimal_idx]
                    
                    cost_data = pd.DataFrame({'Threshold': thresholds, 'Expected Cost': costs})
                    
                    if altair_available():
                        import altair as alt
                        chart = alt.Chart(cost_data).mark_line().encode(
                            x='Threshold:Q', y='Expected Cost:Q'
                        ).properties(title=f'Optimal Threshold: {optimal_threshold:.3f}')
                        
                        # Add optimal point
                        optimal_point = pd.DataFrame({'Threshold': [optimal_threshold], 'Expected Cost': [costs[optimal_idx]]})
                        optimal_chart = alt.Chart(optimal_point).mark_circle(color='red', size=100).encode(
                            x='Threshold:Q', y='Expected Cost:Q'
                        )
                        
                        st.altair_chart(chart + optimal_chart, use_container_width=True)
                    else:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.plot(thresholds, costs)
                        ax.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
                        ax.set_xlabel('Threshold')
                        ax.set_ylabel('Expected Cost')
                        ax.set_title(f'Optimal Threshold: {optimal_threshold:.3f}')
                        ax.legend()
                        st.pyplot(fig)
                    
                    st.info(f"Optimal threshold for cost minimization: {optimal_threshold:.3f}")
                    
                except Exception as e:
                    st.warning(f"Could not generate cost curve: {e}")
        
        else:
            st.warning("Cannot compute analytics: missing required columns for prediction")
    
    # Top Risky Transactions
    if model is not None and target and len(fdf) > 0:
        st.subheader("Top Risky Transactions")
        
        X_risky, _ = safe_select_X(fdf)
        if X_risky is not None:
            # Limit for performance
            max_risky = min(10000, len(X_risky))
            if len(X_risky) > max_risky:
                risky_sample = X_risky.sample(n=max_risky, random_state=42)
                risky_df = fdf.loc[risky_sample.index].copy()
            else:
                risky_df = fdf.copy()
            
            probabilities_risky = proba(model, risky_sample if len(X_risky) > max_risky else X_risky)
            
            if probabilities_risky is not None:
                risky_df['fraud_probability'] = probabilities_risky
                risky_df = risky_df.sort_values('fraud_probability', ascending=False).head(100)
                
                # Hide PII columns by default
                pii_cols = [col for col in risky_df.columns if 'name' in col.lower()]
                show_pii = st.checkbox("Show PII columns", value=False, key="show_pii_checkbox")
                
                display_cols = risky_df.columns.tolist()
                if not show_pii:
                    display_cols = [col for col in display_cols if col not in pii_cols]
                
                st.dataframe(risky_df[display_cols], use_container_width=True)
                
                # Download button
                csv_data = risky_df.to_csv(index=False)
                st.download_button(
                    label="Download Risky Transactions CSV",
                    data=csv_data,
                    file_name="risky_transactions.csv",
                    mime="text/csv"
                )

    st.subheader("Data Preview")
    st.dataframe(fdf.head(20))
    
    # Export buttons - Mobile-friendly
    with st.expander("Export Data", expanded=False):
        csv_data = fdf.to_csv(index=False)
        st.download_button(
            label="Download Filtered Dataset",
            data=csv_data,
            file_name="filtered_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Export metrics as JSON
        if model is not None and target and len(fdf) > 0:
            X_metrics, _ = safe_select_X(fdf)
            if X_metrics is not None:
                try:
                    max_metrics = min(50000, len(X_metrics))
                    if len(X_metrics) > max_metrics:
                        metrics_sample = X_metrics.sample(n=max_metrics, random_state=42)
                        y_metrics = fdf[target].loc[metrics_sample.index]
                    else:
                        metrics_sample = X_metrics
                        y_metrics = fdf[target]
                    
                    probs_metrics = proba(model, metrics_sample)
                    if probs_metrics is not None:
                        y_pred_metrics = (probs_metrics >= threshold).astype(int)
                        
                        from sklearn.metrics import classification_report, confusion_matrix
                        report = classification_report(y_metrics, y_pred_metrics, output_dict=True)
                        cm = confusion_matrix(y_metrics, y_pred_metrics).tolist()
                        
                        metrics_dict = {
                            "threshold": threshold,
                            "total_transactions": len(fdf),
                            "fraud_count": int(fdf[target].sum()),
                            "fraud_rate": float(fdf[target].mean() * 100),
                            "classification_report": report,
                            "confusion_matrix": cm,
                            "timestamp": pd.Timestamp.now().isoformat()
                        }
                        
                        st.download_button(
                            label="Download Metrics JSON",
                            data=json.dumps(metrics_dict, indent=2),
                            file_name="fraud_metrics.json",
                            mime="application/json",
                            use_container_width=True
                        )
                except Exception as e:
                    st.warning(f"Could not generate metrics: {e}")

    # Explainability Section
    if model is not None and target and len(fdf) > 0:
        with st.expander("Model Explainability", expanded=False):
            X_explain, _ = safe_select_X(fdf)
            if X_explain is not None:
                # Feature Importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_names = X_explain.columns
                    importances = model.feature_importances_
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    if altair_available():
                        import altair as alt
                        chart = alt.Chart(importance_df).mark_bar().encode(
                            x=alt.X('Importance:Q', title='Importance'),
                            y=alt.Y('Feature:N', sort='-x', title='Feature')
                        ).properties(height=400)
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 15 Feature Importances')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # SHAP Analysis
                try:
                    import shap
                    
                    st.subheader("SHAP Analysis")
                    
                    # Limit sample for performance
                    max_shap = min(2000, len(X_explain))
                    if len(X_explain) > max_shap:
                        X_shap = X_explain.sample(n=max_shap, random_state=42)
                    else:
                        X_shap = X_explain
                    
                    # Try TreeExplainer first (for tree-based models)
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_shap)
                        
                        # Handle binary classification
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]  # Use fraud class
                        
                        st.subheader("SHAP Summary Plot")
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_shap, show=False)
                        st.pyplot(fig)
                        
                        # SHAP Bar Plot
                        st.subheader("SHAP Feature Importance")
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
                        st.pyplot(fig)
                        
                    except:
                        # Fallback to Explainer
                        explainer = shap.Explainer(model)
                        shap_values = explainer(X_shap)
                        
                        st.subheader("SHAP Summary Plot")
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_shap, show=False)
                        st.pyplot(fig)
                    
                except ImportError:
                    st.info("Install SHAP for advanced explainability")
                    
                    # Fallback to permutation importance
                    try:
                        from sklearn.inspection import permutation_importance
                        
                        st.subheader("Permutation Importance (Fallback)")
                        
                        max_perm = min(2000, len(X_explain))
                        if len(X_explain) > max_perm:
                            X_perm = X_explain.sample(n=max_perm, random_state=42)
                            y_perm = fdf[target].loc[X_perm.index]
                        else:
                            X_perm = X_explain
                            y_perm = fdf[target]
                        
                        perm_importance = permutation_importance(model, X_perm, y_perm, 
                                                               n_repeats=5, random_state=42)
                        
                        perm_df = pd.DataFrame({
                            'Feature': X_perm.columns,
                            'Importance': perm_importance.importances_mean,
                            'Std': perm_importance.importances_std
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        if altair_available():
                            import altair as alt
                            chart = alt.Chart(perm_df).mark_bar().encode(
                                x=alt.X('Importance:Q', title='Permutation Importance'),
                                y=alt.Y('Feature:N', sort='-x', title='Feature')
                            ).properties(height=400)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.barh(perm_df['Feature'], perm_df['Importance'])
                            ax.set_xlabel('Permutation Importance')
                            ax.set_title('Top 15 Permutation Importances')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.warning(f"Could not compute feature importance: {str(e)}")
                        st.info("This may be due to scikit-learn version compatibility issues with the trained model.")

# ---------- Predict ----------
with tab2:
    st.title("Fraud Detection Prediction")
    
    if model is None:
        st.error("Model not loaded. Check the .pkt/.pkl file.")
        st.stop()
    
    # Mobile-friendly prediction form
    st.subheader("Transaction Details")
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"], key="predict_transaction_type")
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=0.01, key="predict_amount")
    oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0, step=0.01, key="predict_oldbalanceOrg")
    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=0.01, key="predict_newbalanceOrig")
    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=0.01, key="predict_oldbalanceDest")
    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=0.01, key="predict_newbalanceDest")
    
    with st.expander("Prediction Settings", expanded=True):
        pred_threshold = st.slider("Decision Threshold", 0.01, 0.99, threshold, 0.01,
                                 help="Probability threshold for fraud classification",
                                 key="predict_threshold_slider")
        
        # Real-time prediction
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])
    
    # Get probability
    prob = proba(model, input_data)
    
    if prob is not None:
        prob_val = float(prob[0])
        pred_val = 1 if prob_val >= pred_threshold else 0
        
        # Display prediction result
        st.subheader("Prediction Result")
        
        if pred_val == 1:
            st.error(f"**FRAUD DETECTED**")
            st.metric("Risk Probability", f"{prob_val*100:.1f}%", 
                     delta=f"Above threshold ({pred_threshold*100:.1f}%)")
        else:
            st.success(f"**SAFE TRANSACTION**")
            st.metric("Risk Probability", f"{prob_val*100:.1f}%", 
                     delta=f"Below threshold ({pred_threshold*100:.1f}%)")
        
        # Risk level indicator
        if prob_val >= 0.8:
            risk_level = "High Risk"
        elif prob_val >= 0.5:
            risk_level = "Medium Risk"
        elif prob_val >= 0.2:
            risk_level = "Low Risk"
        else:
            risk_level = "Very Low Risk"
        
        st.info(f"Risk Level: {risk_level}")
    else:
        st.warning("Could not compute probability")

    # What-if Analysis - Mobile-friendly
    with st.expander("What-If Analysis", expanded=False):
        st.subheader("Scenario Testing")
        
        st.write("**Adjust Amount**")
        whatif_amount = st.slider("Amount", 0.0, 100000.0, amount, 100.0, key="whatif_amount")
        
        st.write("**Adjust Sender Balance**")
        whatif_old_org = st.slider("Old Balance (Sender)", 0.0, 100000.0, oldbalanceOrg, 100.0, key="whatif_old_org")
        whatif_new_org = st.slider("New Balance (Sender)", 0.0, 100000.0, newbalanceOrig, 100.0, key="whatif_new_org")
        
        st.write("**Adjust Receiver Balance**")
        whatif_old_dest = st.slider("Old Balance (Receiver)", 0.0, 100000.0, oldbalanceDest, 100.0, key="whatif_old_dest")
        whatif_new_dest = st.slider("New Balance (Receiver)", 0.0, 100000.0, newbalanceDest, 100.0, key="whatif_new_dest")
        
        st.write("**Transaction Type**")
        whatif_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"], 
                                 index=["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"].index(transaction_type), key="whatif_transaction_type")
        
        # What-if prediction
        whatif_data = pd.DataFrame([{
            "type": whatif_type,
            "amount": whatif_amount,
            "oldbalanceOrg": whatif_old_org,
            "newbalanceOrig": whatif_new_org,
            "oldbalanceDest": whatif_old_dest,
            "newbalanceDest": whatif_new_dest
        }])
        
        whatif_prob = proba(model, whatif_data)
        if whatif_prob is not None:
            whatif_prob_val = float(whatif_prob[0])
            whatif_pred = 1 if whatif_prob_val >= pred_threshold else 0
            
            st.subheader("What-If Result")
            
            if whatif_pred == 1:
                st.error(f"**WOULD BE FRAUD** ({whatif_prob_val*100:.1f}%)")
            else:
                st.success(f"**WOULD BE SAFE** ({whatif_prob_val*100:.1f}%)")
            
            # Compare with original
            if prob is not None:
                prob_change = whatif_prob_val - prob_val
                if prob_change > 0.05:
                    st.warning(f"Risk increased by {prob_change*100:.1f} percentage points")
                elif prob_change < -0.05:
                    st.info(f"Risk decreased by {abs(prob_change)*100:.1f} percentage points")
                else:
                    st.info(f"Risk changed by {prob_change*100:.1f} percentage points")

    # Feature Contribution Analysis
    if prob is not None:
        with st.expander("Feature Contribution Analysis", expanded=False):
            st.subheader("What Drives This Prediction?")
            
            # Get feature contributions if SHAP is available
            try:
                import shap
                
                # Create a small sample for SHAP (just our input)
                explainer = shap.Explainer(model)
                shap_values = explainer(input_data)
                
                # Extract feature values and SHAP values
                feature_names = input_data.columns.tolist()
                feature_values = input_data.iloc[0].values
                shap_vals = shap_values.values[0]
                
                # Create contribution dataframe with consistent data types
                contributions = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': [str(val) for val in feature_values],  # Convert all values to strings
                    'SHAP_Value': shap_vals,
                    'Contribution': shap_vals
                }).sort_values('Contribution', key=abs, ascending=False)
                
                # Color code contributions
                def color_contrib(val):
                    if val > 0:
                        return 'background-color: #ffebee'  # Light red for positive (fraud)
                    else:
                        return 'background-color: #e8f5e8'  # Light green for negative (safe)
                
                styled_contrib = contributions.style.applymap(color_contrib, subset=['Contribution'])
                st.dataframe(styled_contrib, use_container_width=True)
                
                # Top drivers - Mobile-friendly
                st.subheader("Top Risk Drivers")
                top_positive = contributions[contributions['Contribution'] > 0].head(3)
                top_negative = contributions[contributions['Contribution'] < 0].head(3)
                
                st.write("**Pushes toward FRAUD:**")
                for _, row in top_positive.iterrows():
                    st.write(f"• {row['Feature']}: {row['Value']} (+{row['Contribution']:.3f})")
                
                st.write("**Pushes toward SAFE:**")
                for _, row in top_negative.iterrows():
                    st.write(f"• {row['Feature']}: {row['Value']} ({row['Contribution']:.3f})")
                
            except ImportError:
                st.info("Install SHAP for detailed feature contribution analysis")
                
                # Fallback: show feature values
                st.subheader("Feature Values")
                feature_df = pd.DataFrame({
                    'Feature': input_data.columns,
                    'Value': [str(val) for val in input_data.iloc[0].values]  # Convert to strings
                })
                st.dataframe(feature_df, use_container_width=True)

    # Prediction Summary
    st.markdown("---")
    st.subheader("Transaction Summary")
    
    summary_data = input_data.copy()
    if prob is not None:
        summary_data['fraud_probability'] = prob_val
        summary_data['prediction'] = 'FRAUD' if pred_val == 1 else 'SAFE'
        summary_data['threshold_used'] = pred_threshold
    
    st.dataframe(summary_data, use_container_width=True)
    
    # Download prediction result - Mobile-friendly
    with st.expander("Download Results", expanded=False):
        csv_data = summary_data.to_csv(index=False)
        st.download_button(
            label="Download Prediction CSV",
            data=csv_data,
            file_name=f"prediction_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------- Footer ----------
st.markdown(
    '<div class="footer">© '
    + str(pd.Timestamp.now().year)
    + ' · Built with Streamlit by <b>Houda El Barehoumi</b></div>',
    unsafe_allow_html=True
)
