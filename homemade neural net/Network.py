from pybrain.structure import FeedForwardNetwork
Adder = FeedForwardNetwork()
from pybrain.structure import LinearLayer, SigmoidLayer
inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)
from pybrain.structure import FullConnection
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
n.sortModules()
from pybrain.datasets import SupervisedDataSet
basicAdditionPractice = SupervisedDataSet(2, 1)
ds.addSample((0, 0.5), (0.5,))
ds.addSample((0, 0), (0,))
ds.addSample((0.5, 0), (0.5,))
ds.addSample((0.5, 0.5), (1,))