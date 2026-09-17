"""Login / registration page."""
import streamlit as st
from core.auth import register_user, authenticate

def show_auth_page():
    """Display login / registration page."""

    st.markdown("""
        <div style='text-align:center; padding:2rem 0;'>
            <h1 style='background: linear-gradient(90deg, #667eea, #764ba2);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        font-size:3rem; font-weight:900;'>👁️ ARTHA DRISHTI</h1>
            <p style='color:#a0aec0; font-size:1.15rem;'>ML-Based Advanced Stock Screener &amp; Analyzer</p>
        </div>
    """, unsafe_allow_html=True)

    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["🔑 Login", "📝 Sign Up", "👤 Guest Access"])

    with auth_tab1:
        st.markdown("### Login to Your Account")
        with st.form("login_form"):
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            login_submit = st.form_submit_button("Login", type="primary", use_container_width=True)

            if login_submit:
                if not login_user or not login_pass:
                    st.error("Please enter both username and password.")
                else:
                    success, result = authenticate(login_user, login_pass)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.auth_username = login_user.strip().lower()
                        st.session_state.auth_user_data = result
                        # Restore saved watchlist / portfolio
                        st.session_state.watchlist = result.get("watchlist", [])
                        st.session_state['_saved_portfolio'] = result.get("portfolio", [])
                        st.session_state['price_alerts'] = result.get("alerts", [])
                        st.session_state['trade_journal'] = result.get("trade_journal", [])

                        saved_settings = result.get("settings", {}) if isinstance(result.get("settings", {}), dict) else {}
                        if saved_settings:
                            st.session_state.current_exchange = saved_settings.get("default_exchange", st.session_state.get("current_exchange", "NSE"))
                            st.session_state.alert_webhook_url = saved_settings.get("alert_webhook_url", "")
                            st.session_state.enable_email_alerts = bool(saved_settings.get("enable_email_alerts", False))
                            st.session_state.alert_check_interval_sec = int(saved_settings.get("alert_check_interval_sec", 60))
                            st.session_state.alert_repeat_cooldown_min = int(saved_settings.get("alert_repeat_cooldown_min", 60))
                        st.session_state.user_settings_applied = False
                        st.success(f"Welcome back, {result.get('full_name') or login_user}!")
                        st.rerun()
                    else:
                        st.error(result)

    with auth_tab2:
        st.markdown("### Create a New Account")
        with st.form("register_form"):
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_email = st.text_input("Email (optional)", key="reg_email")
            reg_user = st.text_input("Username (min 3 chars)", key="reg_user")
            reg_pass = st.text_input("Password (min 6 chars)", type="password", key="reg_pass")
            reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
            reg_submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if reg_submit:
                if reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(reg_user, reg_pass, reg_name, reg_email)
                    if success:
                        st.success(msg + " You can now log in.")
                    else:
                        st.error(msg)

    with auth_tab3:
        st.markdown("### Continue as Guest")
        st.info("Guest mode gives full access but your watchlist, portfolio, alerts, and trade journal won't be saved between sessions.")
        if st.button("🚀 Continue as Guest", type="primary", use_container_width=True, key="guest_btn"):
            st.session_state.authenticated = True
            st.session_state.auth_username = "__guest__"
            st.session_state.auth_user_data = {}
            st.session_state.watchlist = []
            st.session_state.price_alerts = []
            st.session_state.trade_journal = []
            st.session_state._saved_portfolio = []
            st.session_state.user_settings_applied = False
            st.rerun()


