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
from transformers import pipeline
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_results(y_test, y_pred, df_clean):
    # Plot confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0])
    ax[0].set_title("Confusion Matrix")

    # Plot RiskSegment distribution
    df_clean['RiskSegment'].value_counts().sort_index().plot(
        kind='bar', ax=ax[1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax[1].set_title("Risk Segment Distribution")

def plot_user_cluster_pca(X_cluster, kmeans, user_input, scaler, pca):

    # Scale the features
    X_scaled = scaler.transform(X_cluster)
    # Fit PCA
    X_pca = pca.transform(X_scaled)
    # Get cluster labels
    labels = kmeans.predict(X_scaled)

    # Prepare user input
    user_scaled = scaler.transform([user_input])
    user_pca = pca.transform(user_scaled)

    # Color map for clusters: 0=green (low), 1=orange (medium), 2=red (high)
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    risk_names = {0: 'Low', 1: 'Medium', 2: 'High'}

    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster in range(3):
        idx = labels == cluster
        ax.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            c=colors[cluster], label=f"{risk_names[cluster]} Risk", alpha=0.5, s=30
        )
    # Plot user point
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

    ax[1].set_xlabel("Risk Segment")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

# Load dataset (example: Kaggle's Loan Default Prediction dataset)
url = "loan_data.csv"
df = pd.read_csv(url)

# Basic cleaning: drop rows with missing values
df_clean = df.dropna()

# Encode categorical columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df_clean[col] = le.fit_transform(df_clean[col])

# Drop Loan_ID column if present
if 'Loan_ID' in df_clean.columns:
    df_clean = df_clean.drop('Loan_ID', axis=1)

# Convert Yes/No columns to 1/0
yes_no_cols = [col for col in df_clean.columns if df_clean[col].nunique() == 2 and set(df_clean[col].unique()) == {'Yes', 'No'}]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

# Handle remaining categorical features using one-hot encoding
categorical_cols = df_clean.select_dtypes(include=['object']).columns
df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

# Fill any remaining missing values with column median
df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

print("Preprocessing complete. Data shape:", df_clean.shape)

# Assume the target variable is 'Loan_Status' (adjust if different)
X = df_clean.drop('Default', axis=1)
y = df_clean['Default']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy score:", acc)

cluster_features = ['CreditScore', 'DTIRatio', 'Income', 'LoanAmount']
X_cluster = df_clean[cluster_features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
risk_segments = kmeans.fit_predict(X_scaled)

# Assign RiskSegment labels to the dataframe
df_clean['RiskSegment'] = risk_segments

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("RiskSegment labels assigned. Value counts:")
print(df_clean['RiskSegment'].value_counts())

# Call the function after predictions and after RiskSegment is assigned
plot_results(y_test, y_pred, df_clean)

def predict_loan_default(CreditScore, DTIRatio, Income, LoanAmount, **kwargs):
    # Prepare input for model
    input_dict = {
        'CreditScore': CreditScore,
        'DTIRatio': DTIRatio,
        'Income': Income,
        'LoanAmount': LoanAmount
    }
    # Add other features with default or user-provided values
    for col in X.columns:
        if col not in input_dict:
            input_dict[col] = kwargs.get(col, 0)
    input_df = pd.DataFrame([input_dict])
    # Align columns
    input_df = input_df[X.columns]
    # Predict default
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]
    # Cluster assignment
    scaled = scaler.transform(input_df[cluster_features])
    risk_label = kmeans.predict(scaled)[0]
    risk_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_segment_str = risk_names.get(risk_label, str(risk_label))

    # Recommendations
    recs = []
    if CreditScore < 600:
        recs.append("Improve your credit score for better approval chances.")
    if DTIRatio > 0.4:
        recs.append("Consider reducing your debt-to-income ratio.")
    if Income < df_clean['Income'].median():
        recs.append("Higher income may improve your loan eligibility.")
    if LoanAmount > df_clean['LoanAmount'].median():
        recs.append("Consider applying for a lower loan amount.")
    if not recs:
        recs.append("Your profile looks good for loan approval.")
    
    return (
        "Default" if pred == 1 else "No Default",
        f"Probability of default: {pred_proba:.2f}",
        f"Risk Segment: {risk_segment_str}",
        "\n".join(recs)
    )

# Build Gradio interface
inputs = [
    gr.Number(label="Credit Score"),
    gr.Number(label="DTI Ratio"),
    gr.Number(label="Income"),
    gr.Number(label="Loan Amount")
]
iface = gr.Interface(
    fn=predict_loan_default,
    inputs=inputs,
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability"),
        gr.Textbox(label="Risk Segment"),
        gr.Textbox(label="Recommendations")
    ],
    title="Loan Default Prediction",
    description="Enter your details to get a loan default prediction, risk segment, and recommendations."
)
# Add a natural language input for user profile description

# Load a lightweight LLM for feature extraction (using a zero-shot pipeline for demonstration)
# In production, use a custom model or prompt engineering for better extraction
nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_features_from_text(profile_text):
    features = {}

    # Extract CreditScore
    match = re.search(r"credit score\D*(\d+\.?\d*)", profile_text, re.IGNORECASE)
    if match:
        features["CreditScore"] = float(match.group(1))
    else:
        features["CreditScore"] = 650  # Default value

    # Extract Income - looks for earn/income/salary followed by a number (supports lakh, LPA, etc.)
    match = re.search(r"(earn|income|salary)[^\d]*(\d+\.?\d*)\s*(lpa|lakhs|lakh|k)?", profile_text, re.IGNORECASE)
    if match:
        income_val = float(match.group(2))
        unit = match.group(3)
        # Convert LPA/lakh/k to absolute number (assuming lakhs)
        if unit and unit.lower() in ['lpa', 'lakhs', 'lakh']:
            income_val *= 100000  # convert lakhs to absolute
        elif unit and unit.lower() == 'k':
            income_val *= 1000  # convert thousands to absolute
        features["Income"] = income_val
    else:
        features["Income"] = df_clean['Income'].median()  # Default median income

    # Extract DTI Ratio (supports %, or decimal)
    match = re.search(r"(dti ratio|debt[- ]?to[- ]?income ratio|dti)[^\d]*(\d+\.?\d*)\s*%?", profile_text, re.IGNORECASE)
    if match:
        val = float(match.group(2))
        features["DTIRatio"] = val / 100 if val > 1 else val
    else:
        features["DTIRatio"] = 0.3  # Default DTI

    # Extract LoanAmount (supports lakh/k etc.)
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
        features["LoanAmount"] = df_clean['LoanAmount'].median()  # Default loan amount

    # Extract CoSigner presence
    features["CoSigner"] = 1 if re.search(r"co[- ]?signer|cosigner", profile_text, re.IGNORECASE) else 0

    # You can add more feature extractions here if needed, e.g., Age, NumberOfLoans, Mortgage, Dependents, etc.

    return features

    # Define candidate labels for extraction
    candidate_labels = [
        "CreditScore", "DTIRatio", "Income", "LoanAmount", "Age", "NumberOfLoans", "CoSigner"
    ]
    # Use zero-shot classification to extract relevant features
    result = nlp(profile_text, candidate_labels)
    features = {}
    for label, score in zip(result['labels'], result['scores']):
        # Naive extraction: if label is mentioned, try to extract a number from the text
        match = re.search(rf"{label}\D*(\d+\.?\d*)", profile_text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            features[label] = value
        else:
            # For boolean features like CoSigner
            if label == "CoSigner":
                features[label] = 1 if "co-signer" in profile_text.lower() or "cosigner" in profile_text.lower() else 0
    # Set defaults if not found
    features.setdefault("CreditScore", 650)
    features.setdefault("DTIRatio", 0.3)
    features.setdefault("Income", df_clean['Income'].median())
    features.setdefault("LoanAmount", df_clean['LoanAmount'].median())
    return features

def predict_from_text(profile_text):
    global pca
    features = extract_features_from_text(profile_text)
    
    # Get the prediction details
    prediction, probability, risk_segment, recommendations = predict_loan_default(
        features["CreditScore"],
        features["DTIRatio"],
        features["Income"],
        features["LoanAmount"]
    )
    
    # Prepare user input list for clustering visualization
    user_input = [
        features["CreditScore"],
        features["DTIRatio"],
        features["Income"],
        features["LoanAmount"]
    ]
    
    # Generate the PCA plot with user highlighted
    fig = plot_user_cluster_pca(X_cluster, kmeans, user_input, scaler, pca)
    
    return prediction, probability, risk_segment, recommendations, fig


# Add a new tab for natural language profile input
iface_nl = gr.Interface(
    fn=predict_from_text,
    inputs=gr.Textbox(label="Describe your profile (e.g., 'I earn 5LPA, 25 years old, 2 loans, no co-signer...')"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability"),
        gr.Textbox(label="Risk Segment"),
        gr.Textbox(label="Recommendations"),
        gr.Plot(label="Your Position in Risk Cluster Space")
    ],
    title="Loan Default Prediction (Natural Language)",
    description="Describe your profile in natural language to get a prediction and see where you fall among risk clusters."
)

# Combine both interfaces in a tabbed layout
iface = gr.TabbedInterface([iface, iface_nl], ["Structured Input", "Natural Language Input"])
iface.launch()