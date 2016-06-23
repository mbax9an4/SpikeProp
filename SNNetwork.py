import numpy as np
import math
from Connection import *
from AsyncSN import *


#class to construct the spiking neural network given an array whose length sets the number of layers, 
#and the values are the number of neurons on each layer, without considering the input layer
class SNNetwork:
	def __init__(self, netLayout, inputNeurons, terminals, inhibN):
		noLayers = netLayout.shape

		noInhibN = inhibN

		#will have a size set by noLayers[0]
		self.layers = list()
		for l in range(noLayers[0]):
			neurons = netLayout[l]
			# print 'no of neurons ', neurons
			self.layers.append(np.empty((neurons),dtype=object))

			if l == 0:
				connections = inputNeurons
			else:
				connections = netLayout[l-1]

			for n in range(neurons):
				if l != noLayers[0]:
					self.layers[l][n] = AsyncSN(connections, terminals, noInhibN)
					noInhibN -= 1
				else:
					self.layers[l][n] = AsyncSN(connections, terminals, 0)	

	#returns the last firing times of a layer of neurons
	@classmethod
	def getFireTimesLayer(self, layer):
		noNeurons = layer.shape
		preSNFTime = np.zeros(noNeurons[0])

		for n in range(noNeurons[0]):
			#get the last element of the list storing the firing times of the neuron
			preSNFTime[n] = layer[n].getLastFireTime()

		return preSNFTime

	@classmethod
	def getTypesLayer(self, layer):
		noNeurons = layer.shape
		preSNTypes = np.zeros(noNeurons[0])

		for n in range(noNeurons[0]):
			#get the last element of the list storing the firing times of the neuron
			preSNTypes[n] = layer[n].type

		return preSNTypes


	def resetSpikeTimeNet(self):
		layersNo = len(self.layers)

		for l in range(layersNo):
			noNeurons = self.layers[l].shape
			for n in range(noNeurons[0]):
				self.layers[l][n].resetSpikeTimes()

	def resetSpikeTimeLayer(self, layer):
		noNeurons = layer.shape
		for n in range(noNeurons[0]):
			layer[n].resetSpikeTimes()

	def displaySNN(self):
		print '------------ Displaying the network properties ------------'
		noLayers = len(self.layers)
		for l in range(noLayers):
			neurons = self.layers[l].shape
			print 'Layer ', l, ' has ', neurons[0],' neurons.'
			for n in range(neurons[0]):
				print 'Neuron ', n, ' has the following properties:'
				self.layers[l][n].displaySN()


