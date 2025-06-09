"""Data Sample: 
LoanID,Age,Income,LoanAmount,CreditScore,MonthsEmployed,NumCreditLines,InterestRate,LoanTerm,DTIRatio,Education,EmploymentType,MaritalStatus,HasMortgage,HasDependents,LoanPurpose,HasCoSigner,Default
I38PQUQS96,56,85994,50587,520,80,4,15.23,36,0.44,Bachelor's,Full-time,Divorced,Yes,Yes,Other,Yes,0
HPSK72WA7R,69,50432,124440,458,15,1,4.81,60,0.68,Master's,Full-time,Married,No,No,Other,Yes,0"""
import gradio as gr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("loan_data.csv")

# Filter relevant numerical columns for analysis
numerical_columns = data.select_dtypes(include=np.number).columns

# Function to generate heatmap for correlation
def generate_heatmap():
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[numerical_columns].corr()
    # Ensure values are displayed inside each square and remove lines
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
               square=True, cbar=False, linewidths=0)
    plt.title("Feature Correlation Heatmap")
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
        sns.histplot(data[column], kde=True, bins=30, color="blue")
        plt.title(f"Histogram for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        histograms[column] = plt.gcf()
    return histograms

# Create Gradio app with Blocks instead of Interface
with gr.Blocks(title="Loan Data Analysis Dashboard") as dashboard:
    gr.Markdown("# Loan Data Analysis Dashboard")
    
    # Correlation Heatmap
    gr.Markdown("## Correlation Heatmap")
    heatmap_plot = gr.Plot()
    
    # Statistics Overview
    gr.Markdown("## Statistics Overview")
    stats_df = gr.DataFrame()
    
    # Histograms and Statistics for each variable
    gr.Markdown("## Histograms and Statistics by Variable")
    
    # Dictionary to store all histogram and stat components
    histogram_components = {}
    stat_components = {}
    
    # Create components for each numerical column
    for column in numerical_columns:
        with gr.Accordion(f"Analysis for {column}", open=False):
            with gr.Row():
                histogram_components[column] = gr.Plot(label=f"Histogram for {column}")
                # Use HTML for more aesthetic statistics display
                stat_components[column] = gr.HTML(label=f"Statistics for {column}")
    
    # Function to populate the dashboard
    def populate_dashboard():
        heatmap = generate_heatmap()
        stats = calculate_statistics()
        histograms = generate_histograms()
        
        # Format statistics for each variable with HTML styling
        formatted_stats = {}
        for column in numerical_columns:
            formatted_stats[column] = f"""
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h3 style="color: #2c3e50; margin-top: 0;">{column} Statistics</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 8px; font-weight: bold;">Range:</td><td>{data[column].min()} - {data[column].max()}</td></tr>
                    <tr><td style="padding: 8px; font-weight: bold;">Mean:</td><td>{round(data[column].mean(), 2)}</td></tr>
                    <tr><td style="padding: 8px; font-weight: bold;">Median:</td><td>{round(data[column].median(), 2)}</td></tr>
                    <tr><td style="padding: 8px; font-weight: bold;">Mode:</td><td>{data[column].mode().values[0] if not data[column].mode().empty else "N/A"}</td></tr>
                </table>
            </div>
            """
        
        # Prepare all outputs in the correct order
        outputs = [heatmap, stats]
        for column in numerical_columns:
            outputs.append(histograms[column])
            outputs.append(formatted_stats[column])
        
        return outputs
    
    # Collect all components in the order they will receive values
    all_components = [heatmap_plot, stats_df]
    for column in numerical_columns:
        all_components.append(histogram_components[column])
        all_components.append(stat_components[column])
    
    # Populate dashboard on load
    dashboard.load(populate_dashboard, [], all_components)

# Launch the dashboard
dashboard.launch()