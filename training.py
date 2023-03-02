from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class Train:
    def __init__(self, features, labels, layers, batch_size, epochs, lrate):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.epochs = epochs 
        self.lrate = lrate
        self.layers = layers 

    def startTrain(self):
        self.createModel()
        self.accuracy = self.model.evaluate(self.features, self.labels)
        print('Loss Value: %.4f' % (self.accuracy[0]*100))
        print('Accuracy: %.4f' % (self.accuracy[-1]*100))

    def createModel(self):
        self.model = Sequential()
        # The first hidden layer
        self.model.add(Dense(int(self.layers[1]), input_dim=self.features.shape[1], activation='relu'))
        # Second hidden layer if exists
        if len(self.layers) == 4:
            self.model.add(Dense(int(self.layers[2]), activation='relu'))
        # Third hidden layers if exists
        if len(self.layers) == 5:
            self.model.add(Dense(int(self.layers[2]), activation='relu'))
            self.model.add(Dense(int(self.layers[3]), activation='relu'))
        # Output layer
        self.model.add(Dense(int(self.layers[-1]), activation='softmax'))
        
        print(self.model.summary())

        # Check the number of exclusive classes
        if self.labels.shape[1] == 2:
            self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.lrate), metrics=['accuracy'])
        else : 
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=self.lrate), metrics=['accuracy'])
        
        self.history = self.model.fit(self.features, self.labels, epochs=self.epochs, batch_size=self.batch_size)

    def testModel(self, test_features):
        probabilities = self.model.predict(test_features)
        # Array to store encoded results 
        self.predicts = []
        i = 0
        # One Hot Encoding 
        # "1.0" for value with the highest probability, "0.0" otherwise
        for prob in probabilities:
            max_prob = max(list(prob))
            self.predicts.append([])
            for j in range(0,len(prob)):
                if prob[j] == max_prob:
                    self.predicts[i].append("1.0")
                else: self.predicts[i].append("0.0")
            i += 1
        print(self.predicts)
        
    def plotGraph(self):
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios':[5,5]})
        ax[0].plot(self.history.history['accuracy'])
        ax[0].set_title("Accuracy")
        ax[1].plot(self.history.history['loss'], color='red')
        ax[1].set_title("Loss value")
        plt.subplots_adjust(bottom=0.15)
        fig.supxlabel('Number of Epoches')
        plt.show()
