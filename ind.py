import gradio as gr
from loan import loan_page
from dashboard1 import dashboard

def main():
    with gr.Blocks(title="Loan Management System") as app:
        # Single source of truth for authentication state
        is_authenticated = gr.State(value=False)
        
        with gr.Tabs() as tabs:
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

                def authenticate(passkey, current_state):
                    if passkey == "1234":
                        return True, "✅ Access granted! Switch to the Dashboard tab."
                    else:
                        return current_state, "❌ Invalid passkey. Try again."

                # Connect login button to auth state
                access_button.click(
                    fn=authenticate,
                    inputs=[passkey_input, is_authenticated],
                    outputs=[is_authenticated, access_output]
                )

            # --------- LOAN TAB ----------
            with gr.TabItem("Loan"):
                loan_page.render()

            # --------- DASHBOARD TAB ----------
            with gr.TabItem("Dashboard") as dashboard_tab:
                # Container for conditional content
                with gr.Group():
                    # Auth required message - shown when not authenticated
                    auth_message = gr.Markdown(
                        "🔒 Access Denied. Please enter the passkey on the Home tab.",
                        visible=True
                    )
                    
                    # Dashboard content - hidden initially
                    with gr.Group(visible=False) as dashboard_content:
                        dashboard.render()
                
                # Update dashboard visibility whenever the tab is selected
                def update_dashboard_view(auth_state):
                    if auth_state:
                        return gr.update(visible=False), gr.update(visible=True)
                    else:
                        return gr.update(visible=True), gr.update(visible=False)
                
                # Check auth state when tab is loaded
                dashboard_tab.select(
                    fn=update_dashboard_view,
                    inputs=[is_authenticated],
                    outputs=[auth_message, dashboard_content]
                )
                
            # --------- LOGOUT TAB ----------
            with gr.TabItem("Logout"):
                logout_button = gr.Button("Logout")
                logout_message = gr.Markdown("")
                
                def perform_logout():
                    return False, "You have been logged out."
                
                logout_button.click(
                    fn=perform_logout,
                    inputs=[],
                    outputs=[is_authenticated, logout_message]
                )

    app.launch()

if __name__ == "__main__":
    main()
