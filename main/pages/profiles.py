'''

Created by Callum Bradley

File contains core functionality for webapp regards lightweight disk-based database for storing user Sign-Ups
'''
#import libraries

import streamlit as st
import pandas as pd
import sqlite3

# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB Management
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()

# DB  Functions

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def profilesdisplay():

    st.subheader("User Profiles")
    st.write("N.B. Passwords have been encrypted for User Security")
    user_result = view_all_users()
    clean_db = pd.DataFrame(user_result, columns=["Username", "Password"])
    st.dataframe(clean_db)
