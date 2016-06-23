import numpy as np
from AsyncSN import *
from SNNetwork import *
from DataProc import *
from copy import deepcopy

codingInterval = 26
learningRate = 1
timeStep = 1

inhibN = 1

class SpikeProp:

	#mean squared error function
	@classmethod
	def errorFMSE(self, actualSpike, expSpikeT):
		noNeurons = len(expSpikeT)
		error = 0
		for n in range(noNeurons):
			error += math.pow((actualSpike[n] - expSpikeT[n]),2)
		error /= 2

		return error


	#cross entropy error function
	# def errorFCE(self, ):

	#returns the currentLayer whose spiking times have been updated after the signal has gone through 
	#the layer
	@classmethod
	def forwardPropL(self, prevLayer, currLayer, lNo, time):
		#when there is only one neuron on the layer, an object is returned instead of an array, so 
		#recast it 
		if type(currLayer) is not np.ndarray:
			tmp = [currLayer]
			currLayer = np.asarray(tmp)

		# print 'the current layer is ', currLayer
		noNeurons = currLayer.shape

		#get the number of terminals in the network
		noTerm = currLayer[0].getNTerminals()

		#check on which iteration we are, if we are computing the forward pass of the first layer then use
		#the input data which will just be int or floats, otherwise check for AsyncSN objects
		if isinstance(prevLayer[0], AsyncSN):
			preSNFTime = SNNetwork.getFireTimesLayer(prevLayer)
			preSNTypes = SNNetwork.getTypesLayer(prevLayer)
		else:
			preSNFTime = prevLayer
			preSNTypes = np.ones(prevLayer.size)

		# print 'The presynaptic types and firing times are ', preSNTypes, preSNFTime

		#simulate the passes through the network
		for n in range(noNeurons[0]):
			currLayer[n].actionPot(preSNFTime, time, preSNTypes)
				
		# print 'The updated firing times of the current layer are ', SNNetwork.getFireTimesLayer(currLayer)

		return currLayer


	#function to simulate a forward pass through the network
	@classmethod
	def forwardProp(self, network, inLayer, expSpikes):
		global codingInterval
		global timeStep
		noLayers = len(network.layers)

		# print 'The simulation time is ', time
		time = 0

		while time <= codingInterval:
			#compute the update for the first layer, using the input spikes
			updatedLayer = self.forwardPropL(inLayer, network.layers[0], 0, time)
			network.layers[0] = updatedLayer

			for layer in range(1,noLayers):
				# print 'layer is ', layer, time
				#check for updates for the layers of the network
				updatedLayer = self.forwardPropL(network.layers[layer-1], network.layers[layer], layer, time)
				network.layers[layer] = updatedLayer
			time += timeStep

		predSpikes = SNNetwork.getFireTimesLayer(network.layers[-1])
		return predSpikes

	#method to compute the delta term of the update for the output layer nodes
	@classmethod
	def deltaOutputN(self, neuron, preSNFTimes, preSNTypes, expSpikeT):
		actualSpike = neuron.getLastFireTime()
		error = expSpikeT - actualSpike

		connNo = neuron.getNConnections()
		termNo = neuron.getNTerminals()

		delta = 0.0
		# neuron.displaySN()

		for c in range(connNo):
			if preSNFTimes[c] != -1:
				for t in range(termNo):
					delta += neuron.synapses[c].weights[t] * neuron.termContrDer(preSNFTimes[c], actualSpike, \
							neuron.synapses[c].delays[t], preSNTypes[c])

		return float(error)/delta


	#method to compute the delta term in the update for the hidden layer nodes
	@classmethod
	def deltaHiddenN(self, neuron, preSNFTimes, preSNTypes, nextLayer, deltaNextLayer, currNInd):
		prevError = 0.0
		currError = 0.0
		termSum = 0.0

		actualSpike = neuron.getLastFireTime()

		connNo = neuron.getNConnections()
		termNo = neuron.getNTerminals()

		neuronsNo = nextLayer.shape

		for c in range(connNo):
			if preSNFTimes[c] != -1:
				for t in range(termNo):
					currError += neuron.synapses[c].weights[t] * neuron.termContrDer(preSNFTimes[c], \
									actualSpike, neuron.synapses[c].delays[t], preSNTypes[c])

		for n in range(neuronsNo[0]):
			termNo = nextLayer[n].getNTerminals()
			termSum = 0.0
			for t in range(termNo):
				termSum += nextLayer[n].synapses[currNInd].weights[t] \
							* neuron.termContrDer(actualSpike, nextLayer[n].getLastFireTime(),\
								nextLayer[n].synapses[currNInd].delays[t],\
								neuron.type)

			# print 'delta next layer ', deltaNextLayer[n]
			prevError += termSum * deltaNextLayer[n]

		return prevError / currError


	#compute the gradient approximation around a value for the weights, to be used in gradient checking
	@classmethod
	def approxGradient(self, network, inLayer, expSpikes, neuronInd):
		epsilon = 10**-4

		#simulate the E(w+e) part of the approximation for the ;ast layer of the network
		net = network
		# net.displaySNN()

		connNo = net.layers[-1][neuronInd].getNConnections()
		termNo = net.layers[-1][neuronInd].getNTerminals()
		errPE = np.empty((connNo,termNo))
		errME = np.empty((connNo,termNo))

		for c in range(connNo):
			for t in range(termNo):
				net.resetSpikeTimeNet()
				net.layers[-1][neuronInd].synapses[c].weights[t] += epsilon

				predSpikes = self.forwardProp(net, inLayer, expSpikes)
				# print 'Predicted spikes with + epsilon ', predSpikes
				errPE[c,t] = self.errorFMSE(predSpikes, expSpikes)
				net = network


		#simulate the E(w-e) part of the approximation for the ;ast layer of the network
		net = network
		for c in range(connNo):
			for t in range(termNo):
				net.resetSpikeTimeNet()
				net.layers[-1][neuronInd].synapses[c].weights[t] -= epsilon

				predSpikes = self.forwardProp(net, inLayer, expSpikes)
				print 'Predicted spikes with - epsilon ', predSpikes
				errME[c,t] = self.errorFMSE(predSpikes, expSpikes)
				net = network
		
		print 'error minus is ', errME

		return (errPE - errME) / (2*epsilon)

	#check that the gradeint was computed correctly
	@classmethod
	def checkGradient(self, gradient, approx):
		print '****************************The gradient is '
		print gradient
		print '****************************The approximation is '
		print approx
		print 'Difference between the gradient and its approximation, for a neuron is:'
		print gradient - approx


	#method that modifies the weights of each neuron using the SpikeProp algorithm and gradient descent
	@classmethod
	def backProp(self, network, expSpikeT, inLayer):
		global learningRate

		net = deepcopy(network)

		layersNo = len(network.layers)
		neuronsNo = network.layers[layersNo-1].shape

		deltaNeuron = np.zeros((neuronsNo))

		#compute the update for the output layer and store any variables that might be needed for 
		#the other layers as well
		for n in range(neuronsNo[0]):
			if network.layers[-1][n].getLastFireTime() != -1:
				connNo = network.layers[-1][n].getNConnections()
				termNo = network.layers[-1][n].getNTerminals()

				#array to store the updates for each connection and terminal for the current neuron
				deltaWO = np.zeros((connNo, termNo))

				preSNFTimes = SNNetwork.getFireTimesLayer(network.layers[layersNo-2])
				preSNTypes = SNNetwork.getTypesLayer(network.layers[layersNo-2])

				for c in range(connNo):
					if preSNFTimes[c] != -1:
						for t in range(termNo):
							termContrib =  network.layers[-1][n].termContr(preSNFTimes[c], \
										network.layers[-1][n].getLastFireTime(), \
										network.layers[-1][n].synapses[c].delays[t], \
										preSNTypes[c])

							deltaWO[c,t] = termContrib * (-1) * learningRate
					else:
						deltaWO[c] = np.zeros((termNo))
						# print '------------------', termContrib

				#independent of connection and terminals, store for reuse for hidden layers
				deltaNeuron[n] = self.deltaOutputN(network.layers[-1][n], preSNFTimes, preSNTypes, \
													expSpikeT[n])

				# print 'the delta is ', deltaWO
				# print 'the error is ', deltaNeuron[n]
				updates = np.multiply(deltaWO, deltaNeuron[n])

				net.layers[-1][n].updateWeights(updates)
			else:
				deltaNeuron[n] = 0

			#enable line to do gradient checking
			# self.checkGradient(deltaWO * deltaNeuron[n], \
							# self.approxGradient(network, inLayer, expSpikeT, n))

		#compute the update for the other layers
		for l in range(layersNo-2, -1, -1):
			# print 'backpropagation through layer ', l
			neuronsNo = network.layers[l].shape
			deltaNeuronH = np.zeros((neuronsNo[0]))
			for n in range(neuronsNo[0]):
				if network.layers[l][n].getLastFireTime() != -1:
					connNo = network.layers[l][n].getNConnections()
					termNo = network.layers[l][n].getNTerminals()
					deltaWH = np.zeros((connNo, termNo))

					if l > 0:
						preSNFTimes = network.getFireTimesLayer(network.layers[l-1])
						preSNTypes = SNNetwork.getTypesLayer(network.layers[l-1])
					else:
						preSNFTimes = inLayer
						preSNTypes = np.ones(inLayer.size)

					for c in range(connNo):
						if preSNFTimes[c] != -1:
							for t in range(termNo):
								termContrib = network.layers[l][n].termContr(preSNFTimes[c], \
													network.layers[l][n].getLastFireTime(),\
													network.layers[l][n].synapses[c].delays[t], preSNTypes[c])
		
								deltaWH[c,t] = termContrib * (-1) * learningRate
						else:
							deltaWH[c] = np.zeros((termNo))

					#compute the hidden layer delta
					deltaNeuronH[n] = self.deltaHiddenN(network.layers[l][n], preSNFTimes, preSNTypes,\
														network.layers[l+1], deltaNeuron, n)
					# print 'The delta of neuron ', n,' is ', deltaNeuronH[n]
					# print 'Delta weight hidden are ', deltaWH
					updates = np.multiply(deltaWH, deltaNeuronH[n])
					net.layers[l][n].updateWeights(updates)
				else:
					deltaNeuronH[n] = 0

			deltaNeuron = deltaNeuronH

		return net


	#method to train the neural network, given the training data (inputS) or the input time 
	#sequence of spikes for the input neurons and the expected spiking times (outputS)
	@classmethod
	def train(self, network, inputS, outputS, learningR, epochs):
		global learningRate 
		learningRate = learningR
		lenTimeSeq, inNeurons = inputS.shape

		#this should be done for a number of epochs as well and at the end of each epoch the resetSpikeTimes
		#method should be called

		print '%%%%%%%%%%%%%%%%%%%%Start of simulation%%%%%%%%%%%%%%%%%%%%%%%%'

		for e in range(epochs):
			error = 0
			inputS, outputS = DataProc.shuffleInUnison(inputS, outputS)
			# network.displaySNN()
			#inIndex represents the index of the spikes in the training data time sequence
			for inIndex in range(lenTimeSeq):
				inLayer = inputS[inIndex,:]
				print 'The input layer is ', inLayer
				expSpikes = outputS[inIndex,:]
				print 'The expected spikes are ', expSpikes
				predSpikes = np.zeros((lenTimeSeq))

				print 'The forward propagation phase started *********'
				predSpikes = self.forwardProp(network, inLayer, expSpikes)
				print 'The predicted spike times are ++++++++ ', predSpikes

				scale = 10 if e < 200 else 1

				network = self.backProp(network, expSpikes, inLayer)

				network.resetSpikeTimeNet()

				predSpikes = self.forwardProp(network, inLayer, expSpikes)
				print 'The predicted spike times are ++++++++ ', predSpikes
				# network.displaySNN()
				# network.resetSpikeTimeLayer(network.layers[-1])
				error += self.errorFMSE(expSpikes, predSpikes)

				#the spikes should be reset after each example
				network.resetSpikeTimeNet()

			print 'The error is ', error
			if error == 0:
				break

	# def test(self, ):
