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

# Create Gradio app with Blocks instead of Interface
with gr.Blocks(title="Loan Data Analysis Dashboard", theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("""
    # Loan Data Analysis Dashboard
    
    This dashboard provides comprehensive analysis of loan data, including correlations, 
    statistical summaries, and visualizations of both numerical and categorical variables.
    """)
    
    # Correlation Heatmap
    gr.Markdown("## Correlation Heatmap")
    heatmap_plot = gr.Plot()
    
    # Statistics Overview
    gr.Markdown("## Numerical Statistics Overview")
    stats_df = gr.DataFrame()
    
    # Histograms and Statistics for each numerical variable
    gr.Markdown("## Numerical Variables Analysis")
    
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
    
    # Categorical Variables Analysis
    gr.Markdown("## Categorical Variables Analysis")
    
    # Dictionary to store all pie chart and count components
    pie_components = {}
    count_components = {}
    
    # Create components for each categorical column
    for column in categorical_columns:
        with gr.Accordion(f"Analysis for {column}", open=False):
            with gr.Row():
                pie_components[column] = gr.Plot(label=f"Distribution of {column}")
                # Use HTML for displaying counts in a nicer format
                count_components[column] = gr.HTML(label=f"Counts for {column}")
    
    # Function to populate the dashboard
    def populate_dashboard():
        heatmap = generate_heatmap()
        stats = calculate_statistics()
        histograms = generate_histograms()
        pie_charts, value_counts = generate_pie_charts()
        
        # Format statistics for each numerical variable with HTML styling
        formatted_stats = {}
        for column in numerical_columns:
            formatted_stats[column] = f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 5px solid #3498db;">
                <h3 style="color: #2c3e50; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 10px;">{column} Statistics</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 10px; font-weight: bold; color: #555;">Range:</td><td style="padding: 10px;">{data[column].min()} - {data[column].max()}</td></tr>
                    <tr style="background-color: #f1f1f1;"><td style="padding: 10px; font-weight: bold; color: #555;">Mean:</td><td style="padding: 10px;">{round(data[column].mean(), 2)}</td></tr>
                    <tr><td style="padding: 10px; font-weight: bold; color: #555;">Median:</td><td style="padding: 10px;">{round(data[column].median(), 2)}</td></tr>
                    <tr style="background-color: #f1f1f1;"><td style="padding: 10px; font-weight: bold; color: #555;">Mode:</td><td style="padding: 10px;">{data[column].mode().values[0] if not data[column].mode().empty else "N/A"}</td></tr>
                </table>
            </div>
            """
        
        # Format counts for each categorical variable with HTML styling
        formatted_counts = {}
        for column in categorical_columns:
            count_html = "<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 5px solid #2ecc71;'>"
            count_html += f"<h3 style='color: #2c3e50; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>{column} Value Counts</h3>"
            count_html += "<table style='width: 100%; border-collapse: collapse;'>"
            
            counts = value_counts[column]
            total = sum(counts.values())
            
            for i, (value, count) in enumerate(counts.items()):
                percentage = (count / total) * 100
                bg_color = "#f1f1f1" if i % 2 == 1 else ""
                count_html += f"<tr style='background-color: {bg_color};'>"
                count_html += f"<td style='padding: 10px; font-weight: bold; color: #555;'>{value}:</td>"
                count_html += f"<td style='padding: 10px;'>{count} ({percentage:.1f}%)</td>"
                count_html += "</tr>"
            
            count_html += "</table></div>"
            formatted_counts[column] = count_html
        
        # Prepare all outputs in the correct order
        outputs = [heatmap, stats]
        
        # Add numerical variable outputs
        for column in numerical_columns:
            outputs.append(histograms[column])
            outputs.append(formatted_stats[column])
        
        # Add categorical variable outputs
        for column in categorical_columns:
            outputs.append(pie_charts[column])
            outputs.append(formatted_counts[column])
        
        return outputs
    
    # Collect all components in the order they will receive values
    all_components = [heatmap_plot, stats_df]
    
    # Add numerical variable components
    for column in numerical_columns:
        all_components.append(histogram_components[column])
        all_components.append(stat_components[column])
    
    # Add categorical variable components
    for column in categorical_columns:
        all_components.append(pie_components[column])
        all_components.append(count_components[column])
    
    # Populate dashboard on load
    dashboard.load(populate_dashboard, [], all_components)

# Launch the dashboard
dashboard.launch()
