import tkinter as tk
from tkinter import simpledialog
import subprocess

def select_user_role():
    root = tk.Tk()
    root.withdraw()

    # Prompt the user to select a role
    user_role = simpledialog.askstring("Select User Role", "Choose a user role:\n1. Administrator\n2. Security")

    # Check the selected role and display a message
    if user_role:
        if user_role.lower() == "administrator" or user_role == "1":
            print("Welcome, Administrator!")
            # Launch video_read.py for the Administrator
            subprocess.run(["python", "face_data.py"])
        elif user_role.lower() == "security" or user_role == "2":
            print("Welcome, Security personnel!")
            # Launch face_recognition.py for Security personnel
            subprocess.run(["python", "face__recognition.py"])
        else:
            print("Invalid selection. Please choose between 'Administrator' or 'Security'.")

# Call the function to prompt user role selection
select_user_role()
