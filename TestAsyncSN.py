import numpy as np
from SNNetwork import *
from DataProc import *
from SpikeProp import *

def main():
	data, labels = DataProc.readData('xor.data',2) 

	#add an extra column as the bias 
	data = DataProc.addBias(data)

	learningRate = 0.01
	epochs = 250

	terminals = 16
	inputNeurons = data.shape
	netLayout = np.asarray([5,1])

	#set the number of inhibitory neurons to set in the network
	inhibN = 1
	threshold = 20

	net = SNNetwork(netLayout, inputNeurons[1], terminals, inhibN)
	# net.displaySNN()
	AsyncSN.setThreshold(threshold)
	SpikeProp.train(net, data, labels, learningRate, epochs)
	# net.displaySNN()


#needed in order to be ale to run main 
if __name__ == "__main__":
	main()