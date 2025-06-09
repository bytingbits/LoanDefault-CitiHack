import gradio as gr
from loan import loan_page
from dashboard1 import dashboard

def main():
    with gr.Blocks(title="Loan Management System") as app:
        dashboard_unlocked = gr.State(value=False)

        with gr.Tabs():
            # --------- HOME TAB ----------
            with gr.TabItem("Home"):
                gr.Markdown("""
                # Welcome to the Loan Management System

                Use the tabs to navigate:
                - **Loan** to apply for a loan
                - **Dashboard** to view analytics (requires passkey)
                """)

                with gr.Row():
                    gr.Markdown("### Enter Passkey to access Dashboard:")
                    passkey_input = gr.Textbox(label="Passkey", type="password")
                    access_button = gr.Button("Submit")
                    access_output = gr.Markdown()

                def check_passkey(passkey):
                    if passkey == "1234":
                        return True, "✅ Access granted! Switch to the Dashboard tab."
                    else:
                        return False, "❌ Invalid passkey. Try again."

                access_button.click(
                    fn=check_passkey,
                    inputs=passkey_input,
                    outputs=[dashboard_unlocked, access_output]
                )

            # --------- LOAN TAB ----------
            with gr.TabItem("Loan"):
                loan_page.render()

            # --------- DASHBOARD TAB ----------
            with gr.TabItem("Dashboard"):
                with gr.Blocks() as dashboard_tab:
                    lock_msg = gr.Markdown("🔒 Access Denied. Please enter the passkey on the Home tab.")
                    with gr.Column(visible=False) as dashboard_content:
                        dashboard.render()

                    def unlock_dashboard(allowed):
                        if allowed:
                            return gr.update(visible=False), gr.update(visible=True)
                        else:
                            return gr.update(visible=True), gr.update(visible=False)

                    dashboard_tab.load(
                        fn=unlock_dashboard,
                        inputs=[dashboard_unlocked],
                        outputs=[lock_msg, dashboard_content]
                    )

    app.launch()

if __name__ == "__main__":
    main()
