import tkinter as tk

def noInputOutputLayer():
    tk.messagebox.showinfo(title="Error", message="Input/Output layer cannot be equal to 0.")

def noHidLayer():
    tk.messagebox.showinfo(title="Error", message="No Hidden Layer. Please add at least one.")

def notCreated():
    tk.messagebox.showinfo(title="Error", message="No network created. Please create network.")

def noInputData():
    tk.messagebox.showinfo(title="Error", message="Please load Training Features first.")

def noOutputData():
    tk.messagebox.showinfo(title="Error", message="Please load Training Labels.")

def notTrained():
    tk.messagebox.showinfo(title="Error", message="Network is not trained. Please train network first.")

def noTestData():
    tk.messagebox.showinfo(title="Error", message="Please load Testist Features.")