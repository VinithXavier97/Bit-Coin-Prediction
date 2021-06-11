# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import *
import pandas as pd
df = pd.read_csv("bitcoin_ticker.csv")
dates= df['date_id'].unique()

from tkcalendar import Calendar, DateEntry

root = Tk()
root.title("Tk dropdown example")

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.pack(pady = 100, padx = 100)

# Create a Tkinter variable
tkvar = StringVar(root)

# Dictionary with options
tkvar.set(dates[0]) # set the default option

popupMenu = OptionMenu(mainframe, tkvar, *dates)
Label(mainframe, text="Select Date").grid(row = 1, column = 1)
popupMenu.grid(row = 2, column =1)
text = tk.Text(root)

# on change dropdown value
def change_dropdown(*args):
    date = tkvar.get()
    data_by_date = df[['low','high','volume']][df['date_id'] == date]
    text.insert(tk.END, str(data_by_date))
    text.pack()
    print( data_by_date )

# link function to change dropdown
tkvar.trace('w', change_dropdown)

root.mainloop()

