import gradio as gr
from loan import loan_page
from dashboard1 import dashboard

def index_page():
    with gr.Blocks() as home:
        gr.Markdown("""
        # Welcome to the Loan Management System

        Use the tabs to navigate:
        - **Loan** to apply for a loan
        - **Dashboard** to view summaries
        """)
    return home

with gr.Blocks() as app:
    gr.Markdown("# 🏦 Loan System Interface")
    gr.TabbedInterface(
        [index_page(), loan_page, dashboard],
        tab_names=["Home", "Loan", "Dashboard"]
    )

app.launch()
