import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

sns.set_theme(style='whitegrid')

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🏥 Patient Readmission Prediction")
st.markdown("**Logistic Regression model on simulated UCI Diabetes 130-US Hospitals data**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
n = st.sidebar.slider("Dataset Size (patients)", 1000, 10000, 5000, 500)
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, 5)
random_seed = st.sidebar.number_input("Random Seed", value=42)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Navigation")
sections = ["📊 Dataset Overview", "📈 Exploratory Analysis", "🤖 Model Training", "🎯 Predict a Patient"]
section = st.sidebar.radio("Go to", sections)

# ── Data Generation (cached) ──────────────────────────────────────────────────
@st.cache_data
def generate_data(n, seed):
    np.random.seed(seed)
    age_groups  = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                    '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
    age_weights = [0.01, 0.02, 0.03, 0.05, 0.09, 0.14, 0.22, 0.26, 0.15, 0.03]

    df = pd.DataFrame({
        'age':                np.random.choice(age_groups, n, p=age_weights),
        'gender':             np.random.choice(['Male','Female'], n),
        'time_in_hospital':   np.random.randint(1, 15, n),
        'num_medications':    np.random.randint(1, 40, n),
        'num_lab_procedures': np.random.randint(1, 100, n),
        'num_procedures':     np.random.randint(0, 7, n),
        'number_diagnoses':   np.random.randint(1, 17, n),
        'number_outpatient':  np.random.poisson(0.4, n),
        'number_emergency':   np.random.poisson(0.2, n),
        'number_inpatient':   np.random.poisson(0.6, n),
        'admission_type':     np.random.choice(['Emergency','Elective','Urgent'], n, p=[0.52,0.27,0.21]),
        'discharge_type':     np.random.choice(['Home','Transferred','SNF','Other'], n, p=[0.54,0.10,0.18,0.18]),
        'diabetesMed':        np.random.choice(['Yes','No'], n, p=[0.77,0.23]),
        'insulin':            np.random.choice(['No','Steady','Up','Down'], n, p=[0.47,0.40,0.07,0.06]),
        'A1Cresult':          np.random.choice(['None','>8','>7','Norm'], n, p=[0.60,0.14,0.06,0.20]),
    })

    score = (
        df['number_inpatient'] * 0.5 +
        df['number_emergency'] * 0.4 +
        df['time_in_hospital'] * 0.05 +
        df['number_diagnoses'] * 0.03 +
        (df['discharge_type'] == 'Transferred').astype(int) * 0.6 +
        (df['admission_type'] == 'Emergency').astype(int) * 0.3 +
        np.random.normal(0, 0.3, n)
    )
    df['readmitted'] = (score >= np.percentile(score, 88)).astype(int)
    return df

@st.cache_data
def preprocess_and_train(n, test_size, seed):
    df = generate_data(n, seed)
    df_model = df.copy()

    age_map = {'[0-10)':0,'[10-20)':1,'[20-30)':2,'[30-40)':3,'[40-50)':4,
               '[50-60)':5,'[60-70)':6,'[70-80)':7,'[80-90)':8,'[90-100)':9}
    df_model['age_num'] = df_model['age'].map(age_map)
    df_model['total_visits']        = df_model['number_outpatient'] + df_model['number_emergency'] + df_model['number_inpatient']
    df_model['med_per_day']         = df_model['num_medications'] / df_model['time_in_hospital'].replace(0, 1)
    df_model['high_risk_discharge'] = df_model['discharge_type'].isin(['Transferred','SNF']).astype(int)
    df_model = pd.get_dummies(df_model,
        columns=['gender','admission_type','discharge_type','diabetesMed','insulin','A1Cresult'],
        drop_first=True)
    df_model.drop(columns=['age'], inplace=True)

    X = df_model.drop('readmitted', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df_model['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=seed, stratify=y)

    train_df    = pd.concat([X_train, y_train], axis=1)
    majority    = train_df[train_df['readmitted'] == 0]
    minority    = train_df[train_df['readmitted'] == 1]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=seed)
    train_bal   = pd.concat([majority, minority_up])

    X_train_bal = train_bal.drop('readmitted', axis=1)
    y_train_bal = train_bal['readmitted']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train_scaled, y_train_bal)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return df, model, scaler, X, X_test, y_test, y_pred, y_prob

# Run pipeline
df = generate_data(n, random_seed)
df_result, model, scaler, X, X_test, y_test, y_pred, y_prob = preprocess_and_train(n, test_size, random_seed)
auc = roc_auc_score(y_test, y_prob)

# ── SECTION 1: Dataset Overview ───────────────────────────────────────────────
if section == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{n:,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Readmission Rate", f"{df['readmitted'].mean()*100:.1f}%")
    col4.metric("ROC-AUC Score", f"{auc:.4f}")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ── SECTION 2: EDA ────────────────────────────────────────────────────────────
elif section == "📈 Exploratory Analysis":
    st.header("📈 Exploratory Data Analysis")

    # Class distribution + admission type
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df['readmitted'].value_counts()
    axes[0].bar(['Not Readmitted','Readmitted'], counts.values, color=['#4CAF50','#F44336'])
    axes[0].set_title('Class Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v+20, f'{v} ({v/n*100:.1f}%)', ha='center', fontweight='bold')

    adm_rate = df.groupby('admission_type')['readmitted'].mean() * 100
    axes[1].bar(adm_rate.index, adm_rate.values, color=['#3498db','#e74c3c','#f39c12'])
    axes[1].set_title('Readmission Rate by Admission Type')
    axes[1].set_ylabel('Readmission Rate (%)')
    for i, v in enumerate(adm_rate.values):
        axes[1].text(i, v+0.2, f'{v:.1f}%', ha='center')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Boxplots
    features = ['time_in_hospital', 'num_medications', 'number_diagnoses', 'number_inpatient']
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, feat in zip(axes, features):
        df.boxplot(column=feat, by='readmitted', ax=ax)
        ax.set_title(feat.replace('_',' ').title())
        ax.set_xlabel('Readmitted (0=No, 1=Yes)')
    plt.suptitle('Numeric Features vs Readmission', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Heatmap
    num_cols = ['time_in_hospital','num_medications','num_lab_procedures',
                'number_diagnoses','number_inpatient','number_emergency','readmitted']
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── SECTION 3: Model Results ──────────────────────────────────────────────────
elif section == "🤖 Model Training":
    st.header("🤖 Model Training & Evaluation")

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC Score", f"{auc:.4f}")
    col2.metric("Test Set Size", f"{len(y_test):,}")
    col3.metric("Model", "Logistic Regression")

    st.markdown("---")
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred,
        target_names=['Not Readmitted','Readmitted'], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("---")

    # Confusion Matrix + ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Not Readmitted','Readmitted']).plot(
        ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('Confusion Matrix', fontweight='bold')

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {auc:.3f}')
    axes[1].plot([0,1],[0,1], 'k--', label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Top 15 Feature Importances")
    coef_df = pd.DataFrame({
        'Feature':     X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False).head(15)

    colors = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['Coefficient']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(coef_df['Feature'][::-1], coef_df['Coefficient'][::-1], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Top 15 Feature Importances\n(Red = increases readmission risk | Blue = decreases risk)',
                 fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📋 Project Summary")
    st.info(f"""
    - **Dataset**: {n} patient records  
    - **Model**: Logistic Regression  
    - **ROC-AUC Score**: {auc:.4f}  
    - **Key Findings**:
        - Prior inpatient visits = strongest readmission predictor
        - Emergency admissions carry higher readmission risk
        - High-risk discharge (Transferred/SNF) increases risk
        - More diagnoses correlate with higher readmission chance
    """)

# ── SECTION 4: Predict a Patient ─────────────────────────────────────────────
elif section == "🎯 Predict a Patient":
    st.header("🎯 Predict Readmission for a New Patient")
    st.markdown("Fill in the patient details below and click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Age Group", ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                                          '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'], index=6)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        admission_type = st.selectbox("Admission Type", ['Emergency', 'Elective', 'Urgent'])
        discharge_type = st.selectbox("Discharge Type", ['Home', 'Transferred', 'SNF', 'Other'])

    with col2:
        time_in_hospital   = st.slider("Time in Hospital (days)", 1, 14, 5)
        num_medications    = st.slider("Number of Medications", 1, 40, 15)
        num_lab_procedures = st.slider("Lab Procedures", 1, 100, 45)
        num_procedures     = st.slider("Other Procedures", 0, 6, 1)

    with col3:
        number_diagnoses  = st.slider("Number of Diagnoses", 1, 16, 6)
        number_outpatient = st.slider("Outpatient Visits", 0, 10, 0)
        number_emergency  = st.slider("Emergency Visits", 0, 10, 0)
        number_inpatient  = st.slider("Inpatient Visits", 0, 10, 1)
        diabetesMed       = st.selectbox("Diabetes Medication", ['Yes', 'No'])
        insulin           = st.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])
        A1Cresult         = st.selectbox("A1C Result", ['None', '>8', '>7', 'Norm'])

    if st.button("🔍 Predict Readmission Risk", use_container_width=True):
        age_map = {'[0-10)':0,'[10-20)':1,'[20-30)':2,'[30-40)':3,'[40-50)':4,
                   '[50-60)':5,'[60-70)':6,'[70-80)':7,'[80-90)':8,'[90-100)':9}

        input_dict = {
            'time_in_hospital':   time_in_hospital,
            'num_medications':    num_medications,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures':     num_procedures,
            'number_diagnoses':   number_diagnoses,
            'number_outpatient':  number_outpatient,
            'number_emergency':   number_emergency,
            'number_inpatient':   number_inpatient,
            'age_num':            age_map[age],
            'total_visits':       number_outpatient + number_emergency + number_inpatient,
            'med_per_day':        num_medications / max(time_in_hospital, 1),
            'high_risk_discharge': 1 if discharge_type in ['Transferred','SNF'] else 0,
            'gender_Male':        1 if gender == 'Male' else 0,
            'admission_type_Emergency': 1 if admission_type == 'Emergency' else 0,
            'admission_type_Urgent':    1 if admission_type == 'Urgent' else 0,
            'discharge_type_Other':     1 if discharge_type == 'Other' else 0,
            'discharge_type_SNF':       1 if discharge_type == 'SNF' else 0,
            'discharge_type_Transferred': 1 if discharge_type == 'Transferred' else 0,
            'diabetesMed_Yes':          1 if diabetesMed == 'Yes' else 0,
            'insulin_Steady':           1 if insulin == 'Steady' else 0,
            'insulin_Up':               1 if insulin == 'Up' else 0,
            'insulin_Down':             1 if insulin == 'Down' else 0,
            'A1Cresult_>7':             1 if A1Cresult == '>7' else 0,
            'A1Cresult_>8':             1 if A1Cresult == '>8' else 0,
            'A1Cresult_Norm':           1 if A1Cresult == 'Norm' else 0,
        }

        # Align with training columns
        input_df = pd.DataFrame([input_dict])
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]

        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **High Readmission Risk** — Probability: **{prob*100:.1f}%**")
        else:
            st.success(f"✅ **Low Readmission Risk** — Probability: **{prob*100:.1f}%**")

        # Risk gauge
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([0], [prob], color='#e74c3c' if prob > 0.5 else '#4CAF50', height=0.5)
        ax.barh([0], [1 - prob], left=[prob], color='#eee', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.set_title(f'Risk Score: {prob*100:.1f}%', fontweight='bold')
        st.pyplot(fig)
        plt.close()
