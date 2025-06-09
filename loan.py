import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import gradio as gr
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import requests

def plot_results(y_test, y_pred, df_clean):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0])
    ax[0].set_title("Confusion Matrix")
    df_clean['RiskSegment'].value_counts().sort_index().plot(
        kind='bar', ax=ax[1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax[1].set_title("Risk Segment Distribution")

def plot_user_cluster_pca(X_cluster, kmeans, user_input, scaler, pca):
    user_scaled = scaler.transform([user_input])
    user_pca = pca.transform(user_scaled)
    X_scaled = scaler.transform(X_cluster)
    X_pca = pca.transform(X_scaled)
    labels = kmeans.predict(X_scaled)
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    risk_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster in range(3):
        idx = labels == cluster
        ax.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            c=colors[cluster], label=f"{risk_names[cluster]} Risk", alpha=0.5, s=30
        )
    ax.scatter(
        user_pca[0, 0], user_pca[0, 1],
        c='blue', marker='x', s=200, linewidths=3, label='User'
    )
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("KMeans Clusters (PCA Projection)")
    ax.legend()
    plt.tight_layout()
    return fig

# Load dataset
url = "loan_data.csv"
df = pd.read_csv(url)
df_clean = df.dropna()
categorical_cols = df_clean.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df_clean[col] = le.fit_transform(df_clean[col])
if 'Loan_ID' in df_clean.columns:
    df_clean = df_clean.drop('Loan_ID', axis=1)
yes_no_cols = [col for col in df_clean.columns if df_clean[col].nunique() == 2 and set(df_clean[col].unique()) == {'Yes', 'No'}]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
categorical_cols = df_clean.select_dtypes(include=['object']).columns
df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
print("Preprocessing complete. Data shape:", df_clean.shape)

X = df_clean.drop('Default', axis=1)
y = df_clean['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy score:", acc)

cluster_features = ['CreditScore', 'DTIRatio', 'Income', 'LoanAmount']
X_cluster = df_clean[cluster_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
risk_segments = kmeans.fit_predict(X_scaled)
df_clean['RiskSegment'] = risk_segments
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("RiskSegment labels assigned. Value counts:")
print(df_clean['RiskSegment'].value_counts())
plot_results(y_test, y_pred, df_clean)

def extract_features_from_text(profile_text):
    features = {}
    match = re.search(r"credit score\D*(\d+\.?\d*)", profile_text, re.IGNORECASE)
    features["CreditScore"] = float(match.group(1)) if match else 650
    match = re.search(r"(earn|income|salary)[^\d]*(\d+\.?\d*)\s*(lpa|lakhs|lakh|k)?", profile_text, re.IGNORECASE)
    if match:
        income_val = float(match.group(2))
        unit = match.group(3)
        if unit and unit.lower() in ['lpa', 'lakhs', 'lakh']:
            income_val *= 100000
        elif unit and unit.lower() == 'k':
            income_val *= 1000
        features["Income"] = income_val
    else:
        features["Income"] = df_clean['Income'].median()
    match = re.search(r"(dti ratio|debt[- ]?to[- ]?income ratio|dti)[^\d]*(\d+\.?\d*)\s*%?", profile_text, re.IGNORECASE)
    if match:
        val = float(match.group(2))
        features["DTIRatio"] = val / 100 if val > 1 else val
    else:
        features["DTIRatio"] = 0.3
    match = re.search(r"(loan amount|loan|applying for|loan to)[^\d]*(\d+\.?\d*)\s*(lpa|lakhs|lakh|k)?", profile_text, re.IGNORECASE)
    if match:
        loan_val = float(match.group(2))
        unit = match.group(3)
        if unit and unit.lower() in ['lpa', 'lakhs', 'lakh']:
            loan_val *= 100000
        elif unit and unit.lower() == 'k':
            loan_val *= 1000
        features["LoanAmount"] = loan_val
    else:
        features["LoanAmount"] = df_clean['LoanAmount'].median()
    features["CoSigner"] = 1 if re.search(r"co[- ]?signer|cosigner", profile_text, re.IGNORECASE) else 0
    return features

def predict_loan_default(CreditScore, DTIRatio, Income, LoanAmount, **kwargs):
    input_dict = {
        'CreditScore': CreditScore,
        'DTIRatio': DTIRatio,
        'Income': Income,
        'LoanAmount': LoanAmount
    }
    for col in X.columns:
        if col not in input_dict:
            input_dict[col] = kwargs.get(col, 0)
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[X.columns]
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]
    scaled = scaler.transform(input_df[cluster_features])
    risk_label = kmeans.predict(scaled)[0]
    risk_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_segment_str = risk_names.get(risk_label, str(risk_label))
    income_25 = df_clean['Income'].quantile(0.25)
    income_50 = df_clean['Income'].quantile(0.5)
    income_75 = df_clean['Income'].quantile(0.75)
    loan_25 = df_clean['LoanAmount'].quantile(0.25)
    loan_50 = df_clean['LoanAmount'].quantile(0.5)
    loan_75 = df_clean['LoanAmount'].quantile(0.75)
    recs = []
    if risk_label == 2:
        recs.append("You are in the high-risk segment. Approval is unlikely unless you address the major risk factors below.")
    elif risk_label == 1:
        recs.append("You are in the medium-risk segment. Addressing the following points can improve your chances.")
    else:
        recs.append("You are in the low-risk segment. Your profile is strong, but consider the following for even better terms.")
    if CreditScore < 600:
        recs.append("Your credit score is very low. Focus on paying all bills on time, reducing credit card balances, and avoiding new debt. Consider using a secured credit card to rebuild your score.")
    elif CreditScore < 650:
        recs.append("Your credit score is below average. Try to pay off outstanding debts and avoid late payments to improve your score.")
    elif CreditScore < 700:
        recs.append("Your credit score is fair. Keep making timely payments and avoid taking on unnecessary new credit.")
    else:
        recs.append("Your credit score is good. Keep up the responsible credit behavior and monitor your credit report for errors.")
    if DTIRatio > 0.6:
        recs.append("Your debt-to-income ratio is extremely high (over 60%). Lenders will see this as a major risk. Consider paying down existing debts or increasing your income before applying.")
    elif DTIRatio > 0.4:
        recs.append("Your debt-to-income ratio is above the preferred range. Try to reduce your monthly debt obligations or increase your income.")
    elif DTIRatio > 0.2:
        recs.append("Your debt-to-income ratio is moderate. Keeping it below 20% will help you qualify for better rates.")
    else:
        recs.append("Your debt-to-income ratio is excellent (below 20%). This is a strong point in your application.")
    if Income < income_25:
        recs.append("Your income is in the lower quartile compared to other applicants. Consider applying for a smaller loan or providing proof of stable employment.")
    elif Income < income_50:
        recs.append("Your income is below average. Highlight your job stability or any additional sources of income.")
    elif Income > income_75:
        recs.append("Your income is above average for applicants. This strengthens your application and may help you negotiate better terms.")
    if LoanAmount > loan_75:
        recs.append("You are applying for a loan amount higher than most applicants. Make sure you have strong supporting documents and a clear repayment plan.")
    elif LoanAmount > loan_50:
        recs.append("Your requested loan amount is above average. Ensure your income and credit profile support this amount.")
    elif LoanAmount < loan_25:
        recs.append("Your requested loan amount is modest. This generally improves your approval chances and may help you get a lower interest rate.")
    if CreditScore < 650 and DTIRatio > 0.4:
        recs.append("Both your credit score and debt-to-income ratio are concerning. Focus on improving both before reapplying.")
    if Income < income_25 and LoanAmount > loan_75:
        recs.append("A low income combined with a high loan amount is a red flag for lenders. Consider lowering your loan request or increasing your income.")
    if LoanAmount < 50000 and DTIRatio < 0.2 and CreditScore > 700 and risk_label == 0:
        recs.append("You are applying for a small loan with a strong credit profile and low debt. Approval is very likely. Consider negotiating for a lower interest rate or shorter tenure to save on interest.")
    if DTIRatio > 0.6 and risk_label == 2:
        recs.append("Consider consolidating your debts or seeking financial counseling to manage your obligations.")
    if Income > income_75 and LoanAmount < loan_25 and risk_label == 0:
        recs.append("With your high income and small loan request, you may qualify for pre-approved offers or instant disbursal.")
    return (
        "Default" if pred == 1 else "No Default",
        f"Probability of default: {pred_proba:.2f}",
        f"Risk Segment: {risk_segment_str}",
        "\n".join(recs)
    )


import requests

def refine_paragraph_with_gemini(user_text):
    api_key = "<YOUR API KEY HERE>"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"
    headers = {"Content-Type": "application/json"}
    
    print("Refining paragraph with Gemini API...")
    print(user_text)
    
    prompt = (
        "Rewrite the following paragraph to clearly and concisely describe the applicant's financial profile for a loan application. "
        "Keep all relevant details:\n\n" + user_text
    )
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    params = {"key": api_key}
    
    response = requests.post(url, headers=headers, params=params, json=payload)
    
    print("Response status code:", response.status_code)
    print("Response content:", response.content)
    
    if response.ok:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return user_text  # fallback to original

    
def predict_from_text(profile_text):
    # Refine the paragraph first
    refined_text = refine_paragraph_with_gemini(profile_text)
    print("Refined Text:", refined_text)
    global pca
    # features = extract_features_from_text(profile_text)
    features = extract_features_from_text(refined_text)
    mapped_features_str = "\n".join(f"{k}: {v}" for k, v in features.items())
    prediction, probability, risk_segment, recommendations = predict_loan_default(
        features["CreditScore"],
        features["DTIRatio"],
        features["Income"],
        features["LoanAmount"]
    )
    user_input = [
        features["CreditScore"],
        features["DTIRatio"],
        features["Income"],
        features["LoanAmount"]
    ]
    fig = plot_user_cluster_pca(X_cluster, kmeans, user_input, scaler, pca)
    return mapped_features_str, prediction, probability, risk_segment, recommendations, fig



iface_structured = gr.Interface(
    fn=predict_loan_default,
    inputs=[
        gr.Number(label="Credit Score"),
        gr.Number(label="DTI Ratio"),
        gr.Number(label="Income"),
        gr.Number(label="Loan Amount")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability"),
        gr.Textbox(label="Risk Segment"),
        gr.Textbox(label="Recommendations")
    ],
    title="Loan Default Prediction",
    description="Enter your details to get a loan default prediction, risk segment, and recommendations."
)
iface_nl = gr.Interface(
    fn=predict_from_text,
    inputs=gr.Textbox(label="Describe your profile (e.g., 'I earn 5LPA, 25 years old, 2 loans, no co-signer...')"),
    outputs=[
        gr.Textbox(label="Extracted Feature Mapping"),
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability"),
        gr.Textbox(label="Risk Segment"),
        gr.Textbox(label="Recommendations"),
        gr.Plot(label="Your Position in Risk Cluster Space")
    ],
    title="Loan Default Prediction (Natural Language)",
    description="Describe your profile in natural language to get a prediction and see where you fall among risk clusters."
)
iface = gr.TabbedInterface([iface_structured, iface_nl], ["Structured Input", "Natural Language Input"])
iface.launch()
# loan_page=iface
