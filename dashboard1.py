"""Data Sample: 
LoanID,Age,Income,LoanAmount,CreditScore,MonthsEmployed,NumCreditLines,InterestRate,LoanTerm,DTIRatio,Education,EmploymentType,MaritalStatus,HasMortgage,HasDependents,LoanPurpose,HasCoSigner,Default
I38PQUQS96,56,85994,50587,520,80,4,15.23,36,0.44,Bachelor's,Full-time,Divorced,Yes,Yes,Other,Yes,0
HPSK72WA7R,69,50432,124440,458,15,1,4.81,60,0.68,Master's,Full-time,Married,No,No,Other,Yes,0"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load your dataset
data = pd.read_csv("loan_data.csv")

# Filter relevant columns for analysis
numerical_columns = data.select_dtypes(include=np.number).columns
categorical_columns = [col for col in data.select_dtypes(include=['object']).columns if col != 'LoanID']

# Function to generate heatmap for correlation
def generate_heatmap():
    plt.figure(figsize=(12, 10))
    correlation_matrix = data[numerical_columns].corr()
    # Ensure ALL values are displayed inside each square and remove lines
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
               square=True, cbar=False, linewidths=0, annot_kws={"size": 10})
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

# Function to calculate range, mean, median, and mode
def calculate_statistics():
    stats = []
    for column in numerical_columns:
        stats.append({
            "Attribute": column,
            "Range": f"{data[column].min()} - {data[column].max()}",
            "Mean": round(data[column].mean(), 2),
            "Median": round(data[column].median(), 2),
            "Mode": data[column].mode().values[0] if not data[column].mode().empty else "N/A",
        })
    return pd.DataFrame(stats)

# Function to generate individual histograms for numerical variables
def generate_histograms():
    histograms = {}
    for column in numerical_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[column], kde=True, bins=30, color="#3498db")
        plt.title(f"Distribution of {column}", fontsize=14, fontweight='bold')
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        histograms[column] = plt.gcf()
    return histograms

# Function to generate pie charts for categorical variables
def generate_pie_charts():
    pie_charts = {}
    value_counts = {}
    
    for column in categorical_columns:
        # Count values
        counts = data[column].value_counts()
        value_counts[column] = counts.to_dict()
        
        # Generate pie chart
        plt.figure(figsize=(7, 5))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                shadow=False, startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(counts))))
        plt.axis('equal')
        plt.title(f"Distribution of {column}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        pie_charts[column] = plt.gcf()
    
    return pie_charts, value_counts

# Streamlit UI function
def streamlit_dashboard():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;900&family=Roboto:wght@400;500&display=swap');
        html, body, .main, .stApp {
            font-family: 'Roboto', sans-serif !important;
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%) !important;
            color: #222 !important;
        }
        .top-navbar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            z-index: 9999;
            background: #fff;
            box-shadow: 0 2px 16px #e0e0e0;
            display: flex;
            align-items: center;
            padding: 0.5em 2em;
            height: 64px;
        }
        .nav-logo {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            margin-right: 1.2em;
            box-shadow: 0 0 8px #e0e0e0;
        }
        .nav-link {
            color: #1b5e20 !important;
            font-family: 'Montserrat', sans-serif !important;
            font-size: 1.1rem;
            font-weight: 700;
            margin-right: 2.2em;
            text-decoration: none;
            transition: color 0.2s, border-bottom 0.2s;
            border-bottom: 2px solid transparent;
        }
        .nav-link:hover {
            color: #ff9800 !important;
            border-bottom: 2px solid #ff9800;
        }
        .nav-active {
            color: #ff9800 !important;
            border-bottom: 2px solid #ff9800;
        }
        .stApp { padding-top: 80px !important; }
        .custom-header {
            font-family: 'Montserrat', sans-serif !important;
            font-size: 2.5rem;
            font-weight: 900;
            color: #1b5e20;
            letter-spacing: 1.5px;
            margin-bottom: 0.5em;
        }
        .custom-subheader {
            font-size: 1.15rem;
            color: #ff9800;
            font-weight: 600;
            margin-bottom: 1em;
            font-family: 'Montserrat', sans-serif !important;
        }
        .dashboard-card {
            background: #fff;
            border: 2px solid #e0e0e0;
            box-shadow: 0 0 16px #e0e0e0;
            border-radius: 18px;
            padding: 2em 2em 1.5em 2em;
            margin-bottom: 2em;
            color: #222;
            font-family: 'Roboto', sans-serif !important;
        }
        .stButton>button, .stDownloadButton>button {
            background: linear-gradient(90deg, #43a047 0%, #ff9800 100%);
            color: #fff !important;
            font-size: 1.1rem !important;
            font-family: 'Montserrat', sans-serif !important;
            font-weight: bold !important;
            border-radius: 12px !important;
            padding: 0.9em 2.2em !important;
            margin: 1em 0 1em 0 !important;
            box-shadow: 0 0 12px #ff980044;
            border: none !important;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            transform: scale(1.04);
            box-shadow: 0 0 24px #43a04744, 0 8px 32px #ff980044;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #f8fafc;
            border-radius: 8px;
            border: 1.5px solid #43a047;
            color: #222 !important;
            font-size: 1.1rem;
            font-family: 'Roboto', sans-serif !important;
        }
        .stTextArea>div>textarea {
            background-color: #f8fafc;
            border-radius: 8px;
            border: 1.5px solid #ff9800;
            color: #222 !important;
            font-size: 1.1rem;
            font-family: 'Roboto', sans-serif !important;
        }
        .stMarkdown, .stTitle, .stSubheader {
            color: #1b5e20;
            font-family: 'Montserrat', sans-serif !important;
        }
        .stPlotlyChart, .stPyplot {
            background: #fff !important;
            border-radius: 16px;
            box-shadow: 0 0 16px #e0e0e0;
            padding: 1.7em;
        }
        .dashboard-expander .streamlit-expanderHeader {
            color: #43a047 !important;
            font-family: 'Montserrat', sans-serif !important;
            font-size: 1.1rem;
        }
        .branding-logo {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            margin-bottom: 1em;
            box-shadow: 0 0 8px #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('''
        <div class="top-navbar">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="nav-logo" alt="Loan Brand Logo">
            <a class="nav-link" href="/" target="_self">Home</a>
            <a class="nav-link" href="/loan" target="_self">Loan Prediction</a>
            <a class="nav-link nav-active" href="#" target="_self">Dashboard</a>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="custom-header">📊 Loan Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-subheader">Explore loan data, risk segments, and trends with interactive analytics.</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.header(":blue[Correlation Heatmap]")
    heatmap = generate_heatmap()
    st.pyplot(heatmap)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.header(":blue[Numerical Statistics Overview]")
    stats = calculate_statistics()
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.header(":blue[Numerical Variables Analysis]")
    histograms = generate_histograms()
    for column, fig in histograms.items():
        with st.expander(f"📊 Histogram for {column}"):
            st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.header(":orange[Categorical Variables Analysis]")
    pie_charts, value_counts = generate_pie_charts()
    for column, fig in pie_charts.items():
        with st.expander(f"🧩 Distribution of {column}"):
            st.pyplot(fig)
            st.markdown("**Counts:**")
            counts = value_counts[column]
            for k, v in counts.items():
                st.write(f"{k}: {v}")
    st.markdown('</div>', unsafe_allow_html=True)

# Uncomment the following line to run the Streamlit app
# streamlit_dashboard()