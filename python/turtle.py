Python 2.7.14 (v2.7.14:84471935ed, Sep 16 2017, 20:19:30) [MSC v.1500 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> 
import turtle
 wn = turtle.Screen()
 wn.bgcolor("lightgreen")      # Set the window background color
 wn.title("Hello, Tess!")      # Set the window title

 tess = turtle.Turtle()
 tess.color("blue")            # Tell tess to change her color
 tess.pensize(3)               # Tell tess to set her pen width

 tess.forward(50)
 tess.left(120)
 tess.forward(50)

 wn.mainloop()
