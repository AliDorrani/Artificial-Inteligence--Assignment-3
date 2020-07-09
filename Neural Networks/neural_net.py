import math
import random
import functools
import numpy as np
from utility import alphabetize, abs_mean

import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from copy import deepcopy


class ValuedElement(object):
    """
    This is an abstract class that all Network elements inherit from
    """
    def __init__(self,name,val):
        self.my_name = name
        self.my_value = val

    def set_value(self,val):
        self.my_value = val

    def get_value(self):
        return self.my_value

    def get_name(self):
        return self.my_name

    def __repr__(self):
        return "%s(%1.2f)" %(self.my_name, self.my_value)

class DifferentiableElement(object):
    """
    This is an abstract interface class implemented by all Network
    parts that require some differentiable element.
    """
    def output(self):
        raise NotImplementedError("This is an abstract method")

    def dOutdX(self, elem):
        raise NotImplementedError("This is an abstract method")

    def clear_cache(self):
        """clears any precalculated cached value"""
        pass






class Input(ValuedElement,DifferentiableElement):
    """
    Representation of an Input into the network.
    These may represent variable inputs as well as fixed inputs
    (Thresholds) that are always set to -1.
    """
    def __init__(self,name,val):
        ValuedElement.__init__(self,name,val)
        DifferentiableElement.__init__(self)
    
    
    
    def output(self):
        return self.get_value()
    def dOutdX(self, elem):
        return 0








class Weight(ValuedElement):
    """
    Representation of an weight into a Neural Unit.
    """
    def __init__(self,name,val):
        ValuedElement.__init__(self,name,val)
        self.next_value = None

    def set_next_value(self,val):
        self.next_value = val

    def update(self):
        self.my_value = self.next_value






class Neuron(DifferentiableElement):
    """
    Representation of a single sigmoid Neural Unit.
    """
    def __init__(self, name, inputs, input_weights, use_cache=True):
        assert len(inputs)==len(input_weights)
        for i in range(len(inputs)):
            assert isinstance(inputs[i],(Neuron,Input))
            assert isinstance(input_weights[i],Weight)
        DifferentiableElement.__init__(self)
        self.my_name = name
        self.my_inputs = inputs # list of Neuron or Input instances
        self.my_weights = input_weights # list of Weight instances
        self.use_cache = use_cache
        self.clear_cache()
        self.my_descendant_weights = None
        self.my_direct_weights = None

    def get_descendant_weights(self):
        """
        Returns a mapping of the names of direct weights into this neuron,
        to all descendant weights. For example if neurons [n1, n2] were connected
        to n5 via the weights [w1,w2], neurons [n3,n4] were connected to n6
        via the weights [w3,w4] and neurons [n5,n6] were connected to n7 via
        weights [w5,w6] then n7.get_descendant_weights() would return
        {'w5': ['w1','w2'], 'w6': ['w3','w4']}
        """
        if self.my_descendant_weights is None:
            self.my_descendant_weights = {}
            inputs = self.get_inputs()
            weights = self.get_weights()
            for i in range(len(weights)):
                weight = weights[i]
                weight_name = weight.get_name()
                self.my_descendant_weights[weight_name] = set()
                input = inputs[i]
                if not isinstance(input, Input):
                    descendants = input.get_descendant_weights()
                    for name, s in descendants.items():
                        st = self.my_descendant_weights[weight_name]
                        st = st.union(s)
                        st.add(name)
                        self.my_descendant_weights[weight_name] = st

        return self.my_descendant_weights

    def isa_descendant_weight_of(self, target, weight):
        """
        Checks if [target] is a indirect input weight into this Neuron
        via the direct input weight [weight].
        """
        weights = self.get_descendant_weights()
        if weight.get_name() in weights:
            return target.get_name() in weights[weight.get_name()]
        else:
            raise Exception("weight %s is not connect to this node: %s"
                            %(weight, self))

    def has_weight(self, weight):
        """
        Checks if [weight] is a direct input weight into this Neuron.
        """
        return weight.get_name() in self.get_descendant_weights()

    def get_weight_nodes(self):
        return self.my_weights

    def clear_cache(self):
        self.my_output = None
        self.my_doutdx = {}

    def output(self):
        # Implement compute_output instead!!
        if self.use_cache:
            # caching optimization, saves previously computed output.
            if self.my_output is None:
                self.my_output = self.compute_output()
            return self.my_output
        return self.compute_output()

    def compute_output(self):

        z = 0
        for elem in range(len(self.get_inputs())):
            inp = self.get_inputs()[elem]
            wei = self.get_weights()[elem]
            z+= wei.get_value()*inp.output()
        return 1.0/(1.0 + np.exp(-z))

    def dOutdX(self, elem):
        if self.use_cache:
            if elem not in self.my_doutdx:
                self.my_doutdx[elem] = self.compute_doutdx(elem)
            return self.my_doutdx[elem]
        return self.compute_doutdx(elem)

    def compute_doutdx(self, elem):

        sigDev = (self.output())*(1-self.output())
        weights = self.get_weights

        if (self.has_weight(elem)):
            for i in range(0,len(self.get_inputs())):
                if (self.get_weights()[i] == elem):
                    return (sigDev * (self.get_inputs()[i].output()))
        else :
            inNeurons = self.get_inputs()
            inWeights = self.get_weights()
            dev = 0
            for i in range(len(self.get_weights())):
                if (self.isa_descendant_weight_of(elem, inWeights[i])):
                    input_deriv = self.get_inputs()[i].dOutdX(elem)
                    dev = (sigDev * ((self.get_weights()[i]).get_value()) * (inNeurons[i]).dOutdX(elem))
        return dev

        # raise NotImplementedError("Implement me!")

    def get_weights(self):
        return self.my_weights

    def get_inputs(self):
        return self.my_inputs

    def get_name(self):
        return self.my_name

    def __repr__(self):
        return "Neuron(%s)" %(self.my_name)




class PerformanceElem(DifferentiableElement):
    """
    Representation of a performance computing output node.
    This element contains methods for setting the
    desired output (d) and also computing the final
    performance P of the network.
    This implementation assumes a single output.
    """
    def __init__(self,input,desired_value):
        assert isinstance(input,(Input,Neuron))
        DifferentiableElement.__init__(self)
        self.my_input = input
        self.my_desired_val = desired_value
    def output(self):
        return -0.5*((self.my_desired_val)-(self.my_input.output()))**2


    def dOutdX(self, elem):
        myInput = self.get_input()
        return ((self.my_desired_val - self.my_input.output())*myInput.dOutdX(elem))

    def set_desired(self,new_desired):
        self.my_desired_val = new_desired

    def get_input(self):
        return self.my_input


# class RegularizedPerformanceElem(PerformanceElem):
#     def __init__(self, input, desired_value):
#         if(type(Input) == Neuron):
#             return 0
#         DifferentiableElement.__init__(self)
#         self.my_input = input
#         self.my_desired_val = desired_value
#         self.lambda__ = 0.0001
#         self.weights = None

#     def set_weights(self, Weight):
#         self.weights = Weight

#     def output(self):
#         old_out = -0.5 * ((self.my_desired_val - self.my_input.output()) ** 2)
#         np_w = np.array([item.get_value() for item in self.weights])
#         OutPut = old_out - self.lambda__ * (np.linalg.norm(np_w))
#         return OutPut
        
#     def dOutdX(self, elem):
#         old_dout = (self.my_desired_val - self.my_input.output()) * \
#             self.my_input.dOutdX(elem)
#         dout = old_dout - self.lambda__ * elem.get_value() * 2
#         return dout




class Network(object):
    def __init__(self,performance_node,neurons):
        self.inputs =  []
        self.weights = []
        self.performance = performance_node
        self.output = performance_node.get_input()
        self.neurons = neurons[:]
        self.neurons.sort(key=functools.cmp_to_key(alphabetize))
        for neuron in self.neurons:
            self.weights.extend(neuron.get_weights())
            for i in neuron.get_inputs():
                if isinstance(i,Input) and not ('i0' in i.get_name()) and not i in self.inputs:
                    self.inputs.append(i)
        self.weights.reverse()
        self.weights = []
        for n in self.neurons:
            self.weights += n.get_weight_nodes()

    @classmethod
    def from_layers(self,performance_node,layers):
        neurons = []
        for layer in layers:
            if layer.get_name() != 'l0':
                neurons.extend(layer.get_elements())
        return Network(performance_node, neurons)

    def clear_cache(self):
        for n in self.neurons:
            n.clear_cache()

def seed_random():
    """Seed the random number generator so that random
    numbers are deterministically 'random'"""
    random.seed(0)
    # np.random.seed(0)

def random_weight():
    """Generate a deterministic random weight"""
    # We found that random.randrange(-1,2) to work well emperically 
    # even though it produces randomly 3 integer values -1, 0, and 1.
    return random.randrange(-1, 2)



    # Uncomment the following if you want to try a uniform distribuiton 
    # of random numbers compare and see what the difference is.
    # return random.uniform(-1, 1)

    # When training larger networks, initialization with small, random
    # values centered around 0 is also common, like the line below:
    # return np.random.normal(0,0.1)

def make_neural_net_basic():
    """
    Constructs a 2-input, 1-output Network with a single neuron.
    This network is used to test your network implementation
    and a guide for constructing more complex networks.
    Naming convention for each of the elements:
    Input: 'i'+ input_number
    Example: 'i1', 'i2', etc.
    Conventions: Start numbering at 1.
                 For the -1 inputs, use 'i0' for everything
    Weight: 'w' + from_identifier + to_identifier
    Examples: 'w1A' for weight from Input i1 to Neuron A
              'wAB' for weight from Neuron A to Neuron B
    Neuron: alphabet_letter
    Convention: Order names by distance to the inputs.
                If equal distant, then order them left to right.
    Example:  'A' is the neuron closest to the inputs.
    All names should be unique.
    You must follow these conventions in order to pass all the tests.
    """
    i0 = Input('i0', -1.0) # this input is immutable
    i1 = Input('i1', 0.0)
    i2 = Input('i2', 0.0)

    w1A = Weight('w1A', 1)
    w2A = Weight('w2A', 1)
    wA  = Weight('wA', 1)

    # Inputs must be in the same order as their associated weights
    A = Neuron('A', [i1,i2,i0], [w1A,w2A,wA])
    P = PerformanceElem(A, 0.0)

    # Package all the components into a network
    # First list the PerformanceElem P, Then list all neurons afterwards
    net = Network(P,[A])
    return net

# def make_neural_net_two_layer():
#     i0 = Input('i0', -1.0)
#     i1 = Input('i1', 0)
#     i2 = Input('i2', 0)
#     seed_random()
#     w1A = Weight('w1A', random_weight())
#     w1B = Weight('w1B', random_weight())
#     w2A = Weight('w2A', random_weight())
#     w2B = Weight('w2B', random_weight())
#     wA = Weight('wA', random_weight())
#     wB = Weight('wB', random_weight())
#     wAC = Weight('wAC', random_weight())
#     wBC = Weight('wBC', random_weight())
#     wC = Weight('wAC', random_weight())
#     A = Neuron('A', [i0,i1,i2], [wA,w1A,w2A])
#     B = Neuron('B', [i0,i1,i2], [wB,w1B,w2B])
#     C = Neuron('C', [i0,A,B], [wC,wAC,wBC])
#     P = PerformanceElem(C, 0.0)
#     return Network(P,[A,B,C])


# def make_neural_net_challenging():
    
#     i0 = Input('i0', -1.0)
#     i1 = Input('i1', 0.0)
#     i2 = Input('i2', 0.0)

#     seed_random()
#     w1A = Weight('w1A', random_weight())
#     w1B = Weight('w1B', random_weight())
#     w1C = Weight('w1C', random_weight())
#     w2A = Weight('w2A', random_weight())
#     w2B = Weight('w2B', random_weight())
#     w2C = Weight('w2C', random_weight())

#     wA = Weight('wA', random_weight())
#     wB = Weight('wB', random_weight())
#     wC = Weight('wC', random_weight())
#     wD = Weight('wD', random_weight())
#     wE = Weight('wE', random_weight())

#     wAD = Weight('wAD', random_weight())
#     wAE = Weight('wAE', random_weight())
#     wBD = Weight('wBD', random_weight())
#     wBE = Weight('wBE', random_weight())
#     wCD = Weight('wCD', random_weight())
#     wCE = Weight('wCE', random_weight())
#     wDE = Weight('wDE', random_weight())

#     A = Neuron('A', [i0,i1,i2], [wA,w1A,w2A])
#     B = Neuron('B', [i0,i1,i2], [wB,w1B,w2B])
#     C = Neuron('C', [i0,i1,i2], [wC,w1C,w2C])
#     D = Neuron('D', [i0,A,B,C], [wD,wAD,wBD,wCD])
#     E = Neuron('E', [i0,A,B,C,D], [wE,wAE,wBE,wCE,wDE])

#     P = PerformanceElem(E, 0.0)
#     return Network(P, [A,B,C,D,E])



# def make_neural_net_with_weights():
#     """
#     In this method you are to use the network you designed earlier
#     and set pre-determined weights.  Your goal is to set the weights
#     to values that will allow the "patchy" problem to converge quickly.
#     Your output network should be able to learn the "patchy"
#     dataset within 1000 iterations of back-propagation.
#     """
#     init_weights = { 
#       'wA': 3.810285,
#       'w1A': 3.206646,
#       'w2A': -3.838381,
#       'wB': -3.760194,
#       'w1B': 3.848091,
#       'w2B': -3.245683,
#       'wC': -2.293088,
#       'w1C': -1.519480,
#       'w2C': -1.653193,
#       'wD': 1.808465,
#       'wAD': -2.846714,
#       'wBD': 3.299372,
#       'wCD': 0.558235,
#       'wE': 5.070003,
#       'wAE': -7.328000,
#       'wBE': 6.521601,
#       'wCE': 3.880647,
#       'wDE': 3.975002
#     }

#     return make_net_with_init_weights_from_dict(make_neural_net_challenging,
#                                                 init_weights)

def make_net_with_init_weights_from_dict(net_fn,init_weights):
    net = net_fn()
    for w in net.weights:
        w.set_value(init_weights[w.get_name()])
    return net

def make_net_with_init_weights_from_list(net_fn,init_weights):
    net = net_fn()
    for i in range(len(net.weights)):
        net.weights[i].set_value(init_weights[i])
    return net


def make_neural_net_two_moons():

    i0 = Input('i0', -1.0)  # Bias
    i1 = Input('i1', 0.)
    i2 = Input('i2', 0.)

    seed_random()
    Aweights = {}
    firstLayer = []
    OutPutWeights = []
    for i in range (1,41):
        First_Weight = Weight("w1A1"+str(i),random_weight())
        Sec_Weight = Weight("w2A1"+str(i),random_weight())
        Thrd_Weight = Weight("wA1"+str(i),random_weight())
        firstLayer.append(Neuron("A1" + str(i),[i1,i2,i0],[First_Weight,Sec_Weight,Thrd_Weight]))
    for i in range(1,41):
        OutPutWeights.append(Weight("wA1"+str(i)+"B",random_weight()))
    OutPutWeights.append(Weight("wB",random_weight()))
    B = Neuron("B",firstLayer+[i0],OutPutWeights)
    return Network(PerformanceElem(B,0.0),firstLayer+[B])


def train(network,
          data,      # training data
          rate=1.0,  # learning rate
          target_abs_mean_performance=0.0001,
          max_iterations = 10000,
          verbose=False):
    """Run back-propagation training algorithm on a given network.
    with training [data].   The training runs for [max_iterations]
    or until [target_abs_mean_performance] is reached.
    """
    
    iteration = 0
    while iteration < max_iterations:
        fully_trained = False
        performances = []  # store performance on each data point
        correct = 0
        for datum in data:
            # set network inputs
            for i in range(len(network.inputs)):
                network.inputs[i].set_value(datum[i])

            # set network desired output
            network.performance.set_desired(datum[-1])

            # clear cached calculations
            network.clear_cache()

            result = network.output.output()
            prediction = round(result)

            if prediction == datum[-1]:
                correct += 1

            performance_result = network.performance.output()


            # compute all the weight updates
            for w in network.weights:
                w.set_next_value(w.get_value() +
                                 rate * network.performance.dOutdX(w))

            # set the new weights
            for w in network.weights:
                w.update()

            # save the performance value
            performances.append(network.performance.output())

            # clear cached calculations
            network.clear_cache()

        # compute the mean performance value
        abs_mean_performance = abs_mean(performances)

        if abs_mean_performance < target_abs_mean_performance:
            if verbose:
                print("iter %d: training complete.\n"\
                      "mean-abs-performance threshold %s reached (%1.6f)"\
                      %(iteration,
                        target_abs_mean_performance,
                        abs_mean_performance))
            break

        iteration += 1



        if iteration % 10 == 0 and verbose:
            print("iter %d: mean-abs-performance = %1.6f"\
                  %(iteration,
                    abs_mean_performance))

    print('weights:', network.weights)
    print("Train Acc: ", float(correct)/len(data))
    plot_decision_boundary(network,data)
  


def test(network, data, verbose=False):
    """Test the neural net on some given data."""
    correct = 0
    for datum in data:

        for i in range(len(network.inputs)):
            network.inputs[i].set_value(datum[i])

        # clear cached calculations
        network.clear_cache()
        result = network.output.output()
        prediction = round(result)

        network.clear_cache()

        if prediction == datum[-1]:
            correct+=1
            if verbose:
                print("test(%s) returned: %s => %s [%s]" %(str(datum),
                                                           str(result),
                                                           datum[-1],
                                                           "correct"))
        else:
            if verbose:
                print("test(%s) returned: %s => %s [%s]" %(str(datum),
                                                           str(result),
                                                           datum[-1],
                                                           "wrong"))

    return float(correct)/len(data)





def plot_decision_boundary(network,data, xmin=-10, xmax=10, ymin=-10, ymax=10):
    
    if(input("Do you want to plot ? y/n \n") == 'y'):    
        print("Ploting on duty ...")
        X = np.array([[item[0], item[1]] for item in data])
        y = np.array([item[2] for item in data])
        x_min =  X[:, 0].min() - 10*0.02
        x_max = X[:, 0].max() + 10*0.02
        y_min = X[:, 1].min() - 10*0.02
        y_max = X[:, 1].max() + 10*0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
        temp = np.c_[xx.ravel(), yy.ravel()]
        newdata = []
        for i in range(len(temp)):
            newdata.append((temp[i, 0], temp[i, 1]))
        z = []
        inputSize = len(network.inputs)
        for elem in newdata:
            for i in range(inputSize):
                network.inputs[i].set_value(elem[i])
            network.clear_cache()
            result = network.output.output()
            prediction = round(result)
            network.clear_cache()
            z.append(prediction)
        z = np.array(z)
        z = z.reshape(xx.shape)
        plt.figure(figsize=(6, 6))
        plt.contourf(xx, yy, z,cmap='coolwarm', alpha=1)
        plt.contour(xx, yy, z, colors='gray', linewidths=0.05)
        plt.scatter(X[:, 0], X[:, 1], cmap='binary', edgecolors='black')
        plt.savefig("Graph.png")
        plt.show()

        input('press <ENTER> to continue')


def finite_difference(network):
    weights = list()
    PerfElement = list()
    weights = network.weights
    PerfElement = network.performance

    for weight in weights:
        network.clear_cache()

        preWeight = weight.get_value()
        NewWeight = (weight.get_value() + 1e-8)
        oldPerf = PerfElement.output()
        dev = PerfElement.dOutdx(weight)
        weight.set_value(weight.get_value() + (1e-8))
        network.clear_cache()
        newPer = (network.performance).output()
        weight.set_value(preWeight)
        finite_diff = (newPer - oldPerf) / (1e-8)
        if abs(network.performance.dOutdX(weight) - finite_diff) < 1e-4:
            print("True")
        else:
            print("False")

    network.clear_cache()