import numpy as np

class Variable:
    def __init__(self, label = "", data = 0.0, _children = (), _op = "", trainable = False):
        self.label = label
        self._forward = None
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._trainable = trainable

        # It is important to use list because order of children is important for weight updates
        self._prev = list(_children)
        self._op = _op

    def __str__(self):
        return f"{self.label}:{self.data}"

    def __add__(self, other):
        result = Variable(_children=(self, other), _op="+")

        def forward():
            return self.forward() + other.forward()
        result._forward = forward

        def backward():
            self.grad = 1.0 * result.grad
            other.grad = 1.0 * result.grad
        result._backward = backward
        
        return result

    def __mul__(self, other):
        result = Variable(_children=(self, other), _op="*")

        def forward():
            return self.forward() * other.forward()
        result._forward = forward
        
        def backward():
            self.grad = other.data * result.grad
            other.grad = self.data * result.grad
        result._backward = backward

        return result
    
    def tanh(self):
        result = Variable(_children=(self,), _op="tanh")

        def forward():
            return np.tanh(self.forward())
        result._forward = forward
        
        def backward():
            self.grad = (1 - np.tanh(self.data) ** 2) * result.grad
        result._backward = backward

        return result
    
    def mse(self, target):
        result = Variable(_children=(self, target), _op="mse")

        def forward():
            return (target.forward() - self.forward()) ** 2
        result._forward = forward

        def backward():
            self.grad = 2 * (self.data - target.data)
            target.grad = 2 * (target.data - self.data)
        result._backward = backward

        return result

    def forward(self):
        if self._forward:
            self.data = self._forward()
        return self.data

    def backward(self, learning_rate):
        # Base case
        self.grad = 1.0
        self._backward()

        nodes = list(self._prev)
        while nodes:
            node = nodes.pop()
            nodes.extend(list(node._prev))
            node._backward()
            if (node._trainable):
                node.data = node.data - node.grad * learning_rate

    def update_weights(self, source):
        nodes_current = list(self._prev)
        nodes_source = list(source._prev)
        while nodes_current and nodes_source:
            node_current = nodes_current.pop()
            node_source = nodes_source.pop()

            nodes_current.extend(list(node_current._prev))
            nodes_source.extend(list(node_source._prev))

            if (node_current._trainable):
                node_current.data = node_source.data