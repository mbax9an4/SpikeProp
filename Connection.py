import numpy as np
import math


memPotDecayT = 28
minDelayF = 1
maxDelayF = 6
#class to define the basic Asynchronous Spiking Neuronal model, that will be used as a unit function 
#in networks
#contains constructor for individual neurons, and any function applied to individual neurons
class Connection:

	def __init__(self, terminals, prevLayerN):
		global minDelayF
		global maxDelayF

		#the weights and delays associated with a connection between a presynaptic neuron i and the
		#current neuron j

		wMin = memPotDecayT/(terminals*prevLayerN*self.lifFunction(maxDelayF))
		wMax = memPotDecayT/(terminals*prevLayerN*self.lifFunction(minDelayF))

		self.weights = np.random.uniform(wMin, wMax, terminals)
		self.delays = np.random.uniform(minDelayF, maxDelayF, terminals)

		# self.weights = np.random.uniform(1, 20, terminals)
		# self.delays = np.random.uniform(1, 6, terminals)


	#a standard spike response function describing a postsynaptic potential 
	#creates a leaky-integrate-and-fire neuron
	@classmethod
	def lifFunction(self, time):
		global memPotDecayT

		if time >= 0: 
			div = float(time) / memPotDecayT
			return div * math.exp(1 - div)
		else:
			return 0

	@classmethod
	def normWeights(self, weights):
		print 'before ', weights
		minW = np.amin(weights)
		print minW
		maxW = np.amax(weights)
		print maxW
		out = (weights - float(minW))/(maxW-minW)
		print 'after ', out
		return (weights - minW)/(maxW-minW)