import random
import math
from graphviz import Digraph

# Wrapper class
class Value:
    def __init__(self, data, _children = () , _op = '' , label = '' ):
        self.data = data
        self._prev = _children
        self._op = _op
        self._label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data , _op="+", _children=(self, other) , label=f"{other._label} + {self._label}")

        def _backward() :
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        label = f"{self._label} * {other._label}" if self._label.startswith("w") else f"{other._label} * {self._label}"
        out = Value(self.data * other.data , _op="*", _children=(self, other), label=label)
        def _backward() :
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data - other.data,
            _op="-",
            _children=(self, other)
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out
    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _op="tanh", _children=(self, ))
        def _backward() :
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data ** other.data,
            _op="**",
            _children=(self, other)
        )
        def _backward():
            self.grad += (
                                 other.data * (self.data ** (other.data - 1))
                         ) * out.grad

        out._backward = _backward
        return out

    def __rpow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other ** self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    def __rtruediv__(self, other):
        return other * (self ** -1)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_loop(v):
            if v  not in visited:
                visited.add(v)
                for child in v._prev:
                    build_loop(child)
                topo.append(v)
        build_loop(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data})"


# Single Neuron Class
class Neuron:
    def __init__(self , nin , numL = 1, numCL = 1):

        self.w = [Value(random.uniform(-1, 1) , label=f"w({numL})_{i+1}{numCL}") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1) , label=f"b({numL})_{numCL}")

    def __call__(self, x):
        pre_act = sum((wi * xi for wi , xi in zip(self.w , x) ), self.b)
        out = pre_act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


# A Single Layer of Neurons
class Layer:
    def __init__(self , nin , nout, numL = 1):
        self.neurons = [Neuron(nin, numL, i+1) for i in range(nout)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        params = []
        for neuron in self.neurons:
            pa = neuron.parameters()
            params.extend(pa)
        return params
# MLP
class MLP:
    def __init__(self , nin , nouts):
        sz = [nin ] + nouts
        self.layers = [Layer(sz[i] , sz[i+1] , numL=i+1) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            layer_params = layer.parameters()
            for p in layer_params:
                params.append(p)
        return params

# Testing Neuron MLP
# x = [2,1,-3]
# x = [Value(num, label=f"x{index+1}") for index , num in enumerate(x)]
# n = MLP(3 , [4,4,1])
# out = n(x)
# print(out)

# Creating a Graph
def trace(x:Value):
    nodes, edges = set(), set()
    def build(x: Value):
        if x not in nodes:
            nodes.add(x)
            for child in x._prev:
                edges.add((child, x))
                build(child)

    build(x)
    return nodes, edges

def draw_dot(root):
    root.label = "y_cap"
    nodes, edges = trace(root)
    print(nodes)
    print(edges)
    dot = Digraph(format='svg' , graph_attr={'rankdir':'LR'})
    for node in nodes:
        uid = str(id(node))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, shape = 'record' , label = f"Label: {node._label} |Data: {node.data} | Grad: {node.grad}")
        if node._op:
            dot.node(name = uid + node._op, shape = 'record' , label = f" {node._op}")
            dot.edge(uid + node._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# Building the graph
# dot = draw_dot(out)
# dot.render("graph" , view=True)

# Epoch count
epoch = 5

#Input
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0 , -1.0, -1.0 , 1.0] # Desired targets

## Transforming Scaler to Value Object
for i in range(len(xs)):
    for j in range(len(xs[0])):
        xs[i][j] = Value(xs[i][j] , label=f"x({i}{j})")

# NN Object
n = MLP(3 , [4,4,1])

for k in range(epoch):
    ## Forward Pass
    y_pred = [n(x) for x in xs]
    print(f"For Epoch: {k + 1}", y_pred)
    ## Calculating Loss
    loss = sum(((y_cap - y) ** 2 for y, y_cap in zip(ys, y_pred) ), 0.0)
    print(f"Loss for {k+1}th iteration: {loss.data:.4f}")

    ## Reset the Gradient
    for p in n.parameters():
        p.grad = 0.0

    ## Backward pass
    loss.backward()

    dot = draw_dot(loss)
    dot.render(f"./graph/graph{k+1}" , view=True)

    ## Update the parameters
    for p in n.parameters():
        p.data += -0.05 * p.grad

