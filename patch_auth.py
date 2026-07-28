import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('frontend/streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ── Find boundary markers ──────────────────────────────────────────────────────
auth_start = content.find('# \u2500\u2500 Authentication')
api_start  = content.find('# \u2500\u2500 API helpers')

new_auth_block = '''\
# \u2500\u2500 Authentication \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

# Demo credentials (for hiring company reviewers)
DEMO_USERNAME = "hspecter"
DEMO_PASSWORD = "Test@1234"
DEMO_NAME     = "Harvey Specter"

import bcrypt as _bcrypt


def _build_credentials() -> dict:
    """Fetch all users from DB and build credentials for streamlit-authenticator."""
    users_data = get_all_users() or []
    creds = {"usernames": {}}
    for u in users_data:
        creds["usernames"][u["username"]] = {
            "email": u["email"],
            "name": u["name"],
            "password": u["password_hash"],  # already bcrypt-hashed
        }
    return creds


def _verify_login(username: str, password: str, creds: dict) -> bool:
    """Manually verify username + password against stored bcrypt hash."""
    user = creds["usernames"].get(username)
    if not user:
        return False
    try:
        return _bcrypt.checkpw(password.encode(), user["password"].encode())
    except Exception:
        return False


def _set_authenticated(username: str, creds: dict) -> None:
    """Set the same session_state keys that streamlit-authenticator uses."""
    user = creds["usernames"].get(username, {})
    st.session_state["authentication_status"] = True
    st.session_state["username"] = username
    st.session_state["name"] = user.get("name", username)


def _save_new_user(username: str, name: str, email: str, hashed_pw: str) -> bool:
    """Persist a newly registered user."""
    try:
        from backend.db import supabase, USE_SUPABASE, get_db
        if USE_SUPABASE:
            supabase.table("users").insert({
                "username": username, "name": name,
                "email": email, "password_hash": hashed_pw,
            }).execute()
        else:
            with get_db() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO users (username, name, email, password_hash) "
                    "VALUES (?,?,?,?)",
                    (username, name, email, hashed_pw)
                )
        return True
    except Exception as exc:
        st.error(f"Failed to save user: {exc}")
        return False


# Rebuild credentials on every run (Streamlit re-executes top-to-bottom)
_credentials = _build_credentials()

authenticator = stauth.Authenticate(
    _credentials,
    cookie_name="legal_ai_cookie",
    cookie_key="psl_secure_key_2025",
    cookie_expiry_days=30,
    auto_hash=False,
)

# \u2500\u2500 Show login / register screen when not authenticated \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

if not st.session_state.get("authentication_status"):
    # Hero header
    st.markdown("""
    <div style='text-align:center; padding: 2.5rem 0 0.5rem'>
        <h1 style='font-size:2.6rem; margin-bottom:0.3rem'>\u2696\ufe0f Pearson Specter Litt</h1>
        <p style='color:#aaa; font-size:1.05rem; margin:0'>Legal AI System \u2014 Operator Access</p>
    </div>
    """, unsafe_allow_html=True)

    # Demo banner
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #1a2a4a 0%, #0f1f3d 100%);
        border: 1px solid #2a4a7f;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 1.2rem 0 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    '>
        <span style='font-size:1.5rem'>\U0001f4bc</span>
        <div>
            <div style='color:#7eb8f7; font-weight:600; font-size:0.95rem'>Hiring Manager? Try the demo instantly</div>
            <div style='color:#8899bb; font-size:0.82rem'>Click the button below to auto-fill demo credentials</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # "Use Demo Credentials" button — sets session state so inputs below pre-fill
    if st.button(
        "\U0001f680  Use Demo Credentials",
        use_container_width=True,
        type="secondary",
        key="demo_btn",
    ):
        st.session_state["_demo_username"] = DEMO_USERNAME
        st.session_state["_demo_password"] = DEMO_PASSWORD
        st.rerun()

    st.markdown("---")

    login_tab, register_tab = st.tabs(["\U0001f510 Login", "\U0001f4dd Register"])

    with login_tab:
        # Custom login form so we can control default values
        with st.form("login_form"):
            login_username = st.text_input(
                "Username",
                value=st.session_state.get("_demo_username", ""),
                placeholder="Enter your username",
            )
            login_password = st.text_input(
                "Password",
                type="password",
                value=st.session_state.get("_demo_password", ""),
                placeholder="Enter your password",
            )
            login_btn = st.form_submit_button("\U0001f510 Sign In", use_container_width=True, type="primary")

        if login_btn:
            if _verify_login(login_username, login_password, _credentials):
                _set_authenticated(login_username, _credentials)
                # Clear demo pre-fill from session
                st.session_state.pop("_demo_username", None)
                st.session_state.pop("_demo_password", None)
                st.rerun()
            else:
                st.error("\u274c Username or password is incorrect.")

        if st.session_state.get("_demo_username"):
            st.info(f"\u2139\ufe0f Demo credentials loaded. Click **Sign In** to continue as **{DEMO_NAME}**.")

    with register_tab:
        st.markdown("#### Create a new account")
        with st.form("register_form", clear_on_submit=True):
            reg_name     = st.text_input("Full Name", placeholder="Harvey Specter")
            reg_username = st.text_input("Username", placeholder="hspecter")
            reg_email    = st.text_input("Email", placeholder="harvey@pearsonspecterlitt.com")
            reg_pw       = st.text_input("Password", type="password")
            reg_pw2      = st.text_input("Confirm Password", type="password")
            submitted    = st.form_submit_button("Create Account", use_container_width=True)

        if submitted:
            if not all([reg_name, reg_username, reg_email, reg_pw]):
                st.error("All fields are required.")
            elif reg_pw != reg_pw2:
                st.error("Passwords do not match.")
            elif reg_username in _credentials["usernames"]:
                st.error("Username already exists. Please choose another.")
            else:
                hashed = _bcrypt.hashpw(reg_pw.encode(), _bcrypt.gensalt()).decode()
                if _save_new_user(reg_username, reg_name, reg_email, hashed):
                    st.success(
                        f"\u2705 Account created for **{reg_name}**! "
                        "Switch to the Login tab to sign in."
                    )

    st.stop()

# \u2500\u2500 Authenticated \u2014 sidebar user card + logout \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-user-card'>
        <div class='user-avatar'>\U0001f464</div>
        <div>
            <div class='user-name'>{st.session_state.get('name', 'User')}</div>
            <div class='user-role'>Operator</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Sign Out", key="signout_btn", use_container_width=True):
        for k in ["authentication_status", "username", "name", "_demo_username", "_demo_password"]:
            st.session_state.pop(k, None)
        st.rerun()

'''

new_content = content[:auth_start] + new_auth_block + content[api_start:]

with open('frontend/streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Done. File size delta:", len(new_content) - len(content), "bytes")
