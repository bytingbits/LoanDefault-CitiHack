import streamlit as st
from loan import streamlit_loan_page
from dashboard1 import streamlit_dashboard

def main():
    st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="centered")
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
        .home-card {
            background: #fff;
            border: 2px solid #e0e0e0;
            box-shadow: 0 0 16px #e0e0e0;
            border-radius: 18px;
            padding: 2em 2em 1.5em 2em;
            margin-bottom: 2em;
            color: #222;
            font-family: 'Roboto', sans-serif !important;
        }
        .stButton>button {
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
        .stButton>button:hover {
            transform: scale(1.04);
            box-shadow: 0 0 24px #43a04744, 0 8px 32px #ff980044;
        }
        .stTextInput>div>div>input {
            background-color: #f8fafc;
            border-radius: 8px;
            border: 1.5px solid #43a047;
            color: #222;
            font-size: 1.1rem;
            font-family: 'Roboto', sans-serif !important;
        }
        .stMarkdown, .stTitle, .stSubheader {
            color: #1b5e20;
            font-family: 'Montserrat', sans-serif !important;
        }
        .branding-logo {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            margin-bottom: 1em;
            box-shadow: 0 0 8px #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('''
        <div class="top-navbar">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="nav-logo" alt="Loan Brand Logo">
            <a class="nav-link nav-active" href="#" target="_self">Home</a>
            <a class="nav-link" href="/loan" target="_self">Loan Prediction</a>
            <a class="nav-link" href="/dashboard1" target="_self">Dashboard</a>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="custom-header">🏦 Welcome to Loan Default Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-subheader">AI-powered loan risk analysis, analytics, and recommendations. Secure, fast, and beautiful.</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="home-card">', unsafe_allow_html=True)
        st.markdown("""
        <ul style='font-size:1.1rem;line-height:2.1em;'>
            <li>🔍 <b>Loan Default Prediction</b> – Instantly predict your risk and get actionable recommendations.</li>
            <li>📊 <b>Analytics Dashboard</b> – Explore trends, risk segments, and data visualizations.</li>
            <li>📝 <b>Natural Language Input</b> – Just describe your profile, and our AI will do the rest.</li>
            <li>🔒 <b>Secure & Private</b> – Your data is never stored or shared.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Loan", "Dashboard", "Logout"])
    if page == "Home":
        passkey = st.text_input("Passkey", type="password", key="home_passkey")
        if st.button("🔓 Unlock Dashboard", key="home_unlock_btn"):
            if passkey == "1234":
                st.session_state['authenticated'] = True
                st.success("Access granted! Switch to the Dashboard tab.")
            else:
                st.error("Invalid passkey. Try again.")
    elif page == "Loan":
        streamlit_loan_page()
    elif page == "Dashboard":
        if st.session_state.get('authenticated', False):
            streamlit_dashboard()
        else:
            st.error("🔒 Access Denied. Please enter the passkey on the Home tab.")
    elif page == "Logout":
        st.session_state['authenticated'] = False
        st.success("You have been logged out.")

if __name__ == "__main__":
    main()
