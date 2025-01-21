import streamlit as st
import sqlite3
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()

def init_auth_db():
    conn = sqlite3.connect('etf_users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    
    # Create default admin user if not exists
    default_username = os.getenv('ADMIN_USERNAME', 'admin')
    default_password = os.getenv('ADMIN_PASSWORD', 'admin123')
    
    c.execute('SELECT * FROM users WHERE username = ?', (default_username,))
    if not c.fetchone():
        hashed = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt())
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                 (default_username, hashed))
    
    conn.commit()
    conn.close()

def check_password(username, password):
    conn = sqlite3.connect('etf_users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        stored_password = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)
    return False

def login_user():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if check_password(username, password):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        return False
    
    return True
