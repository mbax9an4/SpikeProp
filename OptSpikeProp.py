import numpy as np
from SNNetwork import *
from DataProc import *
from SpikeProp import *


def main():
	data, labels = DataProc.readData('xor.data',2) 

	#add an extra column as the bias 
	data = DataProc.addBias(data)

	learningRate = 0.1
	epochs = 200

	terminals = 16
	inputNeurons = data.shape
	netLayout = np.asarray([5,1])

	#set the number of inhibitory neurons to set in the network
	inhibN = 0

	print 'This is a test when the spikes are reset for the output layer after each example and for the entire network after each epoch.'

	# for e in range(100,300,50):
	for lr in range(1, 16, 2):
		threshold = 16
		while threshold <= 24:
			learningRate = float(lr)/100
			threshold = float(threshold+0.4)
			print '%%%%%%%%%%%%%%%%A new test iteration is started with epochs ', epochs, ' learning rate ', learningRate, ' and threshold ', threshold
			net = SNNetwork(netLayout, inputNeurons[1], terminals, inhibN)
			AsyncSN.setThreshold(threshold)
			SpikeProp.train(net, data, labels, learningRate, epochs)
	# net.displaySNN()

#needed in order to be ale to run main 
if __name__ == "__main__":
	main()