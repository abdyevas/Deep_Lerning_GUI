from tkinter import *
from gui import GUI
import params

main = Tk()
main.geometry("%sx%s" % (params.WIDTH, params.HEIGHT))
main.title("Neural Network : Graphical representation")
gui = GUI(main)
gui.create_widgets()
gui.pack_widgets()
main.mainloop()