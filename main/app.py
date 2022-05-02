'''

Created by Callum Bradley

File contains functions where application is launched from to intiiate variosu functions of the app

'''
#import libraries

import sqlite3
import streamlit as st
import pandas as pd

# Pages for import
from pages import cleaner as cl # Page for Cleaning dataset
from pages import machinelearning as ml # Page for Machine Learning Single Entry
from pages import machinelearninggroup as mlg # Page for Machine Learning Group Entry
from pages import profiles as prof # Page for displaying User Profiles

# import image for landing page
from PIL import Image
img = Image.open("/images/qub.jpg")


# Security
# passlib,hashlib,bcrypt,scrypt
import hashlib

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
              (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
              (username, password))
    data = c.fetchall()
    return data


def main():
    """
    
    Initiates functionality of the webapp depending on user selection 
    
    """

    st.title("Module Score Predictor App")

    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader(
            "Welcome to the QUB Machine Learning and Module Score Predictor Application. Login to get started!")
              
        # display landing page image
        st.image(img, width = 600)

    # checks if user exists and offers logged in user functionality
    # if not logged in, no access to webapp functions    
    elif choice == "Login":
        st.sidebar.subheader("Existing User")
        
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username, check_hashes(password, hashed_pswd)) 
            if result:

                st.sidebar.success("**Logged In as {}**".format(username))

                task = st.selectbox(
                    "Options", ["1. Clean Dataset", "2a. ML Dataset - Single Entry Prediction", "2b. ML Dataset - Group Entry Prediction", "Profiles"])
                if task == "1. Clean Dataset":
                    cl.cleanerPageDisplay()
                elif task == "2a. ML Dataset - Single Entry Prediction":
                    ml.machinelearningdisplay()
                elif task == "2b. ML Dataset - Group Entry Prediction":
                    mlg.machinelearningdisplaygroup()
                elif task == "Profiles":
                    prof.profilesdisplay()

        else:
            st.sidebar.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.sidebar.subheader("Create New Account")
        st.subheader("Sign Up for instant access!")
        new_user = st.sidebar.text_input("Username")
        new_password = st.sidebar.text_input("Password", type='password')

        if st.sidebar.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")


if __name__ == '__main__':
    main()
