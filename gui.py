import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import params
import error as err
from training import Train

class GUI:
    # Initialize variables, flags
    def __init__(self, main):
        self.root = main
        self.fCreated = False
        self.fTrained = False
        self.fTested = False
        self.fInputData = False
        self.fOutputData = False
        self.fTestData = False

    def create_widgets(self):
        # Main panels 
        self.topPanel = Frame(self.root)
        self.leftPanel = Frame(self.root, width=200, height=params.HEIGHT)
        self.leftSubPanel1 = Frame(self.leftPanel)
        self.leftSubPanel2 = Frame(self.leftPanel)
        self.leftSubPanel3 = Frame(self.leftPanel)
        self.centerPanel = Frame(self.root, width=params.CANVAWIDTH, height=params.CANVAHEIGHT)

        # Menu bar
        self.button1Create = ttk.Button(self.topPanel, width=25, text="Create", command=self.button1)
        self.button2Train = ttk.Button(self.topPanel, width=25, text="Train", command=self.button2)
        self.button3Test = ttk.Button(self.topPanel, width=25, text="Test", command=self.button3)
        self.button4Clear = ttk.Button(self.topPanel, width=25, text="Clear", command=self.button4)
        self.button5Exit = ttk.Button(self.topPanel, width=25, text="Exit", command=self.button5)

        # Left Panel 1 widgets
        self.inputLabel = tk.Label(self.leftSubPanel1, text="Input layer")
        self.inputCombo = ttk.Combobox(self.leftSubPanel1, values=[1,2,3,4,5])
        self.inputCombo.current(2)
        self.outputLabel = tk.Label(self.leftSubPanel1, text="Output layer")
        self.outputCombo = ttk.Combobox(self.leftSubPanel1, values=[1,2,3,4,5])
        self.outputCombo.current(2)
        self.hidLabel1 = tk.Label(self.leftSubPanel1, text="Hidden layer 1")
        self.hidCombo1 = ttk.Combobox(self.leftSubPanel1, values=[1,2,3,4,5,6,7,8,9,10])
        self.hidCombo1.current(5)
        self.hidLabel2 = tk.Label(self.leftSubPanel1, text="Hidden layer 2")
        self.hidCombo2 = ttk.Combobox(self.leftSubPanel1, values=[0,1,2,3,4,5,6,7,8,9,10])
        self.hidCombo2.current(0)
        self.hidLabel3 = tk.Label(self.leftSubPanel1, text="Hidden layer 3")
        self.hidCombo3 = ttk.Combobox(self.leftSubPanel1, values=[0,1,2,3,4,5,6,7,8,9,10])
        self.hidCombo3.current(0)
        self.hidButton1 = ttk.Button(self.leftSubPanel1, text="Add Hidden layer", command = self.add_layer2, width=20)
        self.hidButton2 = ttk.Button(self.leftSubPanel1, text="Add Hidden layer", command = self.add_layer3, width=20)
        self.buttonCreateNetwork = tk.Button(self.leftSubPanel1, text="Create Network", command=self.createNetwork, width=18)

        # Left Panel 2 widgets
        self.trainLabel = tk.Label(self.leftSubPanel2, text="Training", pady=10)
        self.epochsLabel = tk.Label(self.leftSubPanel2, text="Nb of Epochs")
        self.epochsCombo = ttk.Combobox(self.leftSubPanel2, values=[10,50,100,500,1000,5000,10000,50000,100000])
        self.epochsCombo.current(2)
        self.batchLabel = tk.Label(self.leftSubPanel2, text="Batch size")
        self.batchCombo = ttk.Combobox(self.leftSubPanel2, values=[16,32,64,128,256,512])
        self.batchCombo.current(1)
        self.lrateLabel = tk.Label(self.leftSubPanel2, text="Learning rate")
        self.lrateCombo = ttk.Combobox(self.leftSubPanel2, values=[0.1,0.05,0.01,0.005,0.001,0.0005,0.0001])
        self.lrateCombo.current(4)
        self.loadInputs = tk.Button(self.leftSubPanel2, text="Load Training Features", command=self.loadInputs, width=18)
        self.inputFile = tk.Label(self.leftSubPanel2, text="")
        self.loadOutputs = tk.Button(self.leftSubPanel2, text="Load Training Labels", command=self.loadOutputs, width=18)
        self.outputFile = tk.Label(self.leftSubPanel2, text="")
        self.trainButton = tk.Button(self.leftSubPanel2, text="Train", command=self.trainNetwork, width=18)
        self.lossValue = tk.Label(self.leftSubPanel2, text="Loss Value and Accuracy")
        self.lossBox = tk.Text(self.leftSubPanel2, width=21, height=4)
        self.lossBox.config(state="disabled")
        self.showGraph = tk.Button(self.leftSubPanel2, text="Show graph", command=self.createGraph, width=18)

        # Left Panel 3 widgets
        self.testLabel = tk.Label(self.leftSubPanel3, text="Testing")
        self.testData = tk.Button(self.leftSubPanel3, text="Load Testing Features", command=self.loadTest, width=18)
        self.testFile = tk.Label(self.leftSubPanel3, text="")
        self.testButton = tk.Button(self.leftSubPanel3, text="Predict", command=self.testNetwork, width=18)
        self.testBox = tk.Text(self.leftSubPanel3, width=21, height=10)
        self.testScrollBar = tk.Scrollbar(self.leftSubPanel3, orient=VERTICAL)
        self.testBox.config(yscrollcommand=self.testScrollBar.set, state="disabled")
        self.testScrollBar.config(command=self.testBox.yview)

        # Center widgets
        self.canvas = Canvas(self.centerPanel, bg="white", width=params.CANVAWIDTH, height=params.CANVAHEIGHT, highlightbackground="black", highlightthickness=3)

        # Array to store input values to create network
        self.dataCombobox1 = []
        self.dataCombobox1.append(self.inputCombo)
        self.dataCombobox1.append(self.hidCombo1)
        self.dataCombobox1.append(self.hidCombo2)
        self.dataCombobox1.append(self.hidCombo3)
        self.dataCombobox1.append(self.outputCombo)

        # Array to store training inputs
        self.trainParams = []
        self.trainParams.append(self.epochsCombo)
        self.trainParams.append(self.batchCombo)
        self.trainParams.append(self.lrateCombo)

    def pack_widgets(self):
        # packing widgets
        self.topPanel.pack(side=TOP, pady=(20,0))
        self.leftPanel.pack(side=LEFT, padx=(15,0), fill=X, expand=FALSE)
        self.leftSubPanel1.pack(side=LEFT)
        self.centerPanel.pack(side=LEFT, padx=20, pady=20)

        self.button1Create.pack(side=LEFT, padx=2)
        self.button2Train.pack(side=LEFT, padx=2)
        self.button3Test.pack(side=LEFT,padx=2)
        self.button4Clear.pack(side=LEFT, padx=2)
        self.button5Exit.pack(side=LEFT, padx=2)

        self.inputLabel.pack(side=TOP)
        self.inputCombo.pack(side=TOP)
        self.outputLabel.pack(side=TOP)
        self.outputCombo.pack(side=TOP)
        self.hidLabel1.pack(side=TOP)
        self.hidCombo1.pack(side=TOP)
        self.hidButton1.pack(side=TOP, pady=5)
        self.buttonCreateNetwork.pack(side=BOTTOM, pady=20)

        self.trainLabel.pack(side=TOP)
        self.epochsLabel.pack(side=TOP)
        self.epochsCombo.pack(side=TOP)
        self.batchLabel.pack(side=TOP)
        self.batchCombo.pack(side=TOP)
        self.lrateLabel.pack(side=TOP)
        self.lrateCombo.pack(side=TOP)
        self.loadInputs.pack(side=TOP, pady=(15,5))
        self.inputFile.pack(side=TOP)
        self.loadOutputs.pack(side=TOP, pady=5)
        self.outputFile.pack(side=TOP)
        self.trainButton.pack(side=TOP, pady=15)
        self.showGraph.pack(side=BOTTOM, pady=10)
        self.lossValue.pack(side=TOP)
        self.lossBox.pack(side=LEFT)

        self.testLabel.pack(side=TOP)
        self.testData.pack(side=TOP, pady=5)
        self.testFile.pack(side=TOP)
        self.testButton.pack(side=TOP, pady=(25,15))
        self.testBox.pack(side=LEFT)
        self.testScrollBar.pack(side=LEFT, fill=Y)
    
        self.canvas.pack(side=TOP)

    # Create button
    def button1(self):
        self.leftSubPanel1.pack(side=LEFT, fill=BOTH)
        self.leftSubPanel2.pack_forget()
        self.leftSubPanel3.pack_forget()

    # Train button 
    def button2(self):
        self.leftSubPanel2.pack(side=LEFT, fill=BOTH)
        self.leftSubPanel1.pack_forget()
        self.leftSubPanel3.pack_forget()

    # Test button
    def button3(self):
        self.leftSubPanel3.pack(side=LEFT, fill=BOTH)
        self.leftSubPanel1.pack_forget()
        self.leftSubPanel2.pack_forget()

    # Clear button
    def button4(self):
        self.clear()
        self.inputFile["text"] = ""
        self.outputFile["text"] = ""
        self.testFile["text"] = ""
        self.dataCombobox1[0] = self.inputCombo
        self.dataCombobox1[-1] = self.outputCombo
        if self.fTrained:
            self.lossBox.config(state="normal")
            self.lossBox.delete(1.0, END)
            self.lossBox.config(state="disabled")
        if self.fTested:
            self.testBox.config(state="normal")
            self.testBox.delete(1.0, END)
            self.testBox.config(state="disabled")
        self.fCreated = False
        self.fInputData = False
        self.fOutputData = False
        self.fTrained = False
        self.fTestData = False
        self.fTested = False

    # Exit button
    def button5(self):
        self.root.destroy()

    def remove_layer2(self):
        self.hidLabel2.pack_forget()
        self.hidCombo2.pack_forget()
        self.hidCombo2.current(0)
        self.hidButton1.configure(text="Add Hidden layer", command=self.add_layer2)
        self.hidButton2.pack_forget()

    def add_layer2(self):
        self.hidLabel2.pack(side = TOP)
        self.hidCombo2.pack(side = TOP)
        self.hidButton1.pack_forget()
        self.hidButton2.pack(side = TOP, pady=5)
        self.hidButton1.pack(side = TOP, pady=5)
        self.hidButton1.configure(text="Remove Hidden layer", command=self.remove_layer2)

    def remove_layer3(self):
        self.hidLabel3.pack_forget()
        self.hidCombo3.pack_forget()
        self.hidCombo3.current(0)
        self.hidButton2.pack_forget()
        self.hidButton2.pack(side = TOP, pady=5)
        self.hidButton2.configure(text="Add Hidden layer", command=self.add_layer3)
        self.hidButton1.pack(side = TOP, pady=5)

    def add_layer3(self):
        self.hidLabel3.pack(side = TOP)
        self.hidCombo3.pack(side = TOP)
        self.hidButton1.pack_forget()
        self.hidButton2.pack_forget()
        self.hidButton2.pack(side = TOP, pady=5)
        self.hidButton2.configure(text="Remove Hidden layer", command=self.remove_layer3)

    def createNetwork(self):
        # Array to store input values; zeros removed 
        self.dataCombobox2 = []
        # Arrays to store node coordinates for each layer
        self.inputCoord = []
        self.outputCoord = []
        self.hiddenCoord1 = []
        self.hiddenCoord2 = []
        self.hiddenCoord3 = []
        nbLayers = 0
        if self.inputCombo.get() == '0' or self.outputCombo.get() == '0':
            err.noInputOutputLayer()
        elif self.hidCombo1.get() == '0' and self.hidCombo2.get() == '0' and self.hidCombo3.get() == '0':
            err.noHidLayer()
        else:
            if self.fCreated == True: 
                self.clear()  
            # Check if the input and output nodes number were changes to the ones in the data files
            if isinstance(self.dataCombobox1[0], str) and isinstance(self.dataCombobox1[-1], str):
                self.dataCombobox2.append(self.dataCombobox1[0])
                nbLayers += 1
                for layer in self.dataCombobox1[1:-1]:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        nbLayers += 1
                self.dataCombobox2.append(self.dataCombobox1[-1])
                nbLayers += 1  
            # Check if only the input nodes number were changed 
            elif isinstance(self.dataCombobox1[0], str):
                self.dataCombobox2.append(self.dataCombobox1[0])
                nbLayers += 1
                for layer in self.dataCombobox1[1:]:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        nbLayers += 1
            # Check if only the output nodes number were changed 
            elif isinstance(self.dataCombobox1[-1], str):
                for layer in self.dataCombobox1[:-1]:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        nbLayers += 1
                self.dataCombobox2.append(self.dataCombobox1[-1])
                nbLayers += 1
            else:
                for layer in self.dataCombobox1:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        nbLayers += 1
            print("Nodes in", nbLayers, "layers:", self.dataCombobox2)
            self.createNode(nbLayers)
            self.createArrows()
              
    def createNode(self, nblayers):
        distX = params.CANVAWIDTH / (nblayers + 1) 
        i = 1        
        for nodes in self.dataCombobox2:
            for node in range(1,int(nodes) + 1):
                x = i * distX + params.offsetX
                y = node * (params.CANVAHEIGHT / (int(nodes) + 1)) + params.offsetY
                self.canvas.create_oval(x, y, x + params.nodeWidth, y + params.nodeWidth, fill="blue")
                # Append input layer coordinates
                if i == 1: self.inputCoord.append([x,y])
                # Append first hidden layer coordinates
                elif i == 2: self.hiddenCoord1.append([x,y])
                # Append either output coordinates, or second hidden layer 
                elif i == 3: 
                    if i == nblayers:
                        self.outputCoord.append([x,y])
                    else: self.hiddenCoord2.append([x,y])
                # Append either output coordinates, or third hidden layer 
                elif i == 4: 
                    if i == nblayers:
                        self.outputCoord.append([x,y])
                    else: self.hiddenCoord3.append([x,y])
                # Append output layer coordinates
                elif i == 5: self.outputCoord.append([x,y])
            i += 1
        self.fCreated = True
        # Add input values from the data file to the nodes
        if self.fInputData:
            j = 0
            for coord in self.inputCoord:
                self.canvas.create_text(coord[0] - params.offsetText, coord[1] + params.offsetText, text=self.train[j][0], fill="black", tag="drag")
                j += 1
        # Add output values from the data file to the nodes
        if self.fOutputData:
            k = 0
            for coord in self.outputCoord:
                self.canvas.create_text(coord[0] - params.offsetText, coord[1] + params.offsetText, text=self.y_train[k][0], fill="black", tag="drag")
                k += 1
     
    def createArrows(self):
        shift = params.nodeWidth
        # Create arrow between each node pair
        for cin in self.inputCoord:
            for ch1 in self.hiddenCoord1:
                self.canvas.create_line(cin[0] + shift, cin[1] + shift/2, ch1[0], ch1[1] + shift/2, width=2)
                if self.hiddenCoord2: 
                    for ch2 in self.hiddenCoord2:
                        self.canvas.create_line(ch1[0] + shift, ch1[1] + shift/2, ch2[0], ch2[1] + shift/2, width=2)
                        if self.hiddenCoord3:
                            for ch3 in self.hiddenCoord3:
                                self.canvas.create_line(ch2[0] + shift, ch2[1] + shift/2, ch3[0], ch3[1] + shift/2, width=2)
        for cout in self.outputCoord:
            if self.hiddenCoord3:
                for ch3 in self.hiddenCoord3:
                    self.canvas.create_line(ch3[0] + shift, ch3[1] + shift/2, cout[0], cout[1] + shift/2, width=2)
            elif self.hiddenCoord2:
                for ch2 in self.hiddenCoord2:
                    self.canvas.create_line(ch2[0] + shift, ch2[1] + shift/2, cout[0], cout[1] + shift/2, width=2)
            else: 
                for ch1 in self.hiddenCoord1:
                    self.canvas.create_line(ch1[0] + shift, ch1[1] + shift/2, cout[0], cout[1] + shift/2, width=2)
            
    def clear(self):
        self.canvas.delete("all")

    def loadInputs(self):
        if self.fCreated == False:
            err.notCreated()
        else:
            self.pInputData = ""
            self.pInputData = askopenfilename(initialdir="./data/", filetypes=[("Text file", "*.txt"), ("CSV Files","*.csv")], title="Choose Input Data.")
            self.train = pd.read_csv(self.pInputData, sep=',', header=None)
            filename = self.pInputData.split('/')[len(self.pInputData.split('/'))-1]
            # Add filename to the GUI
            self.inputFile["text"] = filename
            self.fInputData = True
            # Change input nodes number to the node number in the data file
            self.dataCombobox1[0] = str(self.train.shape[1])
            self.createNetwork()

    def loadOutputs(self):
        if self.fCreated == False:
            err.notCreated()
        elif self.fInputData == False:
            err.noInputData()
        else:
            self.pOutputData = ""
            self.pOutputData = askopenfilename(initialdir="./data/", filetypes=[("Text file", "*.txt"), ("CSV Files","*.csv")], title="Choose Input Data.")
            self.y_train = pd.read_csv(self.pOutputData, sep=',', header=None)
            filename = self.pOutputData.split('/')[len(self.pOutputData.split('/'))-1]
            # Add filename to GUI
            self.outputFile["text"] = filename
            self.fOutputData = True
            # Change output nodes number to the node number in the data file 
            self.dataCombobox1[-1] = str(self.y_train.shape[1])
            self.createNetwork()

    def trainNetwork(self):
        if self.fCreated == False:
            err.notCreated()
        elif self.fInputData == False:
            err.noInputData()
        elif self.fOutputData == False:
            err.noOutputData()
        else:
            if self.fTrained:
                self.lossBox.config(state="normal")
                self.lossBox.delete(1.0, END)
            self.epoch = int(self.trainParams[0].get())
            self.batch_size = int(self.trainParams[1].get())
            self.lrate = float(self.trainParams[2].get())
            # Import Train class from training.py file
            self.training = Train(self.train, self.y_train, self.dataCombobox2, self.batch_size, self.epoch, self.lrate)
            self.training.startTrain()
            self.fTrained = True
            # Show training results in the GUI textbox
            self.lossBox.config(state="normal")
            self.lossBox.insert(1.0, self.training.accuracy[0])
            self.lossBox.insert(END, "\n")
            self.lossBox.insert(END, self.training.accuracy[1])
            self.lossBox.config(state="disabled")
    
    def createGraph(self):
        if self.fTrained == False:
            err.notTrained()
        else:
            self.training.plotGraph()

    def loadTest(self):
        if self.fTrained == False:
            err.notTrained()
        else:
            self.pTestData = ""
            self.pTestData = askopenfilename(initialdir="./data/", filetypes=[("Text file", "*.txt"), ("CSV Files","*.csv")], title="Choose Input Data.")
            self.test = pd.read_csv(self.pTestData, sep=',', header=None)
            filename = self.pTestData.split('/')[len(self.pTestData.split('/'))-1]
            # Add filename to GUI
            self.testFile["text"] = filename
            self.fTestData = True         

    def testNetwork(self):
        if self.fCreated == False:
            err.notCreated()
        elif self.fTrained == False:
            err.notTrained()
        elif self.fTestData == False:
            err.noTestData()
        else:
            if self.fTested:
                self.testBox.config(state="normal")
                self.testBox.delete(1.0, END)
            self.training.testModel(self.test)
            self.fTested = True
            # Show predicted outputs in the GUI textbox
            self.testBox.config(state="normal")
            list_pred = self.training.predicts
            for pred in list_pred:
                self.testBox.insert(END, pred)
                self.testBox.insert(END, "\n")
            self.testBox.config(state="disabled")