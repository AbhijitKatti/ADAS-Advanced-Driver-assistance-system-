import sys
import os
from tkinter import *

window=Tk()

heading = Label(window,text="ATRIA INSTITUTE OF TECHNOLOGY",font=('arial 13 bold'),fg="steelblue").pack()
entr_amt=Label(window,text="GROUP 14 ADVANCED DRIVER ASSISTANCE SYSTEM",font=('arial 13 bold'),fg="steelblue").pack()
ent_amt=Label(window,text="STUDENTS\n PRERAN M KYASVAR (1AT18IS065)\nSAHANA K C (1AT18IS076)\nSAHANA K H (1AT18IS077)",font=('arial 13 bold'),fg="steelblue").place(x=250,y=300)

window.title("PROJECT DEMONSTRATION")
window.geometry('550x400')

def run_lane():
    os.system('python lane.py')

def run_Traffic():
    os.system('python traffic_signal.py')



btn = Button(window, text="Lane",width=  12 ,height = 3, bg="blue", fg="yellow",command=run_lane,font=('arial 17 bold') ).place(x=60,y=125)
btn1 = Button(window, text="Traffic",width=  12 ,height = 3, bg="blue", fg="yellow",command=run_Traffic,font=('arial 17 bold')).place(x=300,y=125)



window.mainloop()

