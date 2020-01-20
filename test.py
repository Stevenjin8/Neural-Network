import numpy as np
import gzip, cPickle
from mlxtend.data import loadlocal_mnist
import random
from PIL import Image

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def backprop(self, x, y, lmbda = 0):
        nabla_b = [np.zeros(b.shape) for b in net.biases]
        nabla_w = [np.zeros(w.shape) for w in net.weights]
        activation = x
        activations=[x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range( 2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)



    def cost_derivative(self, output_activations, y):
        a = output_activations
        return (a-y)*np.reciprocal(a*(1-a))

    def cost_derivative2(self, output_activations, y):

        a = []
        for z,Y in zip(output_activations,y):
            if Y==1 and z>=1:
                a.append([0])
            elif Y==1 and z<1:
                a.append([-1])
            elif Y==0 and z<=-1:
                a.append([0])
            elif Y==0 and z>-1:
                a.append([1])
        a= np.array(a)
        return a

    def update_mini_batch(self, mini_batch, eta, lmbda = 0):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        l = len(mini_batch)
        for x,y in mini_batch:
            db, dw = self.backprop(x,y)
            nabla_w = [d+w for d, w in zip(dw, nabla_w)]
            nabla_b = [d+b for d, b in zip(db, nabla_b)]
        self.weights = [(1-lmbda*eta/50000)*w - eta*d/l for w, d in zip(self.weights, nabla_w)]
        self.biases = [b - eta*d/l for b, d in zip(self.biases, nabla_b)]


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda = lmbda)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def eval2(self, data):
        out = []
        for x in range(0,len(data)):
            if not np.argmax(self.feedforward(data[x][0])) == data[x][1]:
                out.append(x)
        return out

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

net = Network([784,100,10])

train_images, train_labels = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')

test_images, test_labels = loadlocal_mnist(
        images_path='test-images-idx3-ubyte',
        labels_path='test-labels-idx1-ubyte')

val_images = train_images[-10001:-1]
val_labels = train_labels[-10001:-1]
train_images = train_images[0:-10000]
train_labels = train_labels[0:-10000]

train_labels = np.array([[0]*x + [1] + [0]*(9-x) for x in train_labels])*1.

test_data = ([(np.reshape(x, (784, 1))/255. , y) for x,y in zip(test_images, test_labels)])
train_data = ([(np.reshape(x, (784, 1))/255. , np.reshape(y, (10,1))) for x,y in zip(train_images, train_labels)])
val_data = ([(np.reshape(x, (784, 1))/255. , y) for x,y in zip(val_images, val_labels)])

net = Network([784,100,10])

def create_image(arr,w):
    o = []
    for x in range(0,len(arr), w):
        p = []
        for y in arr[x:x+w]:
            p.append((y,y,y))
        o.append(p)
    new_image = Image.fromarray(np.array(o))
    new_image.save('new.png')
