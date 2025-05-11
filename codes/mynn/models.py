"""
custom_nn_layers.py

Complete implementation of neural network building blocks and model definitions:
- Layer abstraction
- Linear (fully-connected)
- conv2D (2D convolution with padding & stride)
- ReLU activation
- Softmax + Cross-entropy loss
- L1/L2 regularization
- Flatten & Reshape helpers
- Model_MLP and Model_CNN with load/save
"""

from abc import ABC, abstractmethod
import numpy as np
import pickle

# -----------------------------------------------------------------------------
# Base Layer
# -----------------------------------------------------------------------------
class Layer(ABC):
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    def __call__(self, x):
        return self.forward(x)

# -----------------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------------
class Linear(Layer):
    def __init__(self, in_dim, out_dim,
                 initialize_method=np.random.normal,
                 weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': np.zeros_like(self.W),
                      'b': np.zeros_like(self.b)}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def forward(self, X):
        self.input = X
        return X.dot(self.W) + self.b

    def backward(self, grad):
        # grad: [batch, out_dim]
        self.grads['W'] = self.input.T.dot(grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        # return gradient w.r.t. input: [batch, in_dim]
        return grad.dot(self.W.T)

    def clear_grad(self):
        self.grads = {'W': np.zeros_like(self.W),
                      'b': np.zeros_like(self.b)}

# -----------------------------------------------------------------------------
# 2D Convolution
# -----------------------------------------------------------------------------
class conv2D(Layer):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal,
                 weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels,
                                         in_channels,
                                         kernel_size,
                                         kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W': np.zeros_like(self.W),
                      'b': np.zeros_like(self.b)}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def forward(self, X):
        batch, _, H, W = X.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        # output dims
        H_out = (H + 2*p - k)//s + 1
        W_out = (W + 2*p - k)//s + 1
        # pad input
        X_pad = np.pad(X,
                       ((0,0),(0,0),(p,p),(p,p)),
                       mode='constant')
        out = np.zeros((batch, self.out_channels, H_out, W_out))
        for b in range(batch):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h0, w0 = i*s, j*s
                        window = X_pad[b, :, h0:h0+k, w0:w0+k]
                        out[b, oc, i, j] = np.sum(window * self.W[oc]) + self.b[0, oc, 0, 0]
        self.input = X_pad
        return out

    def backward(self, grads):
        X_pad = self.input
        batch, _, H_pad, W_pad = X_pad.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        _, _, H_out, W_out = grads.shape

        dX_pad = np.zeros_like(X_pad)
        self.grads['W'].fill(0)
        self.grads['b'].fill(0)

        for b in range(batch):
            for oc in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h0, w0 = i*s, j*s
                        window = X_pad[b, :, h0:h0+k, w0:w0+k]
                        self.grads['W'][oc] += window * grads[b, oc, i, j]
                        self.grads['b'][0, oc, 0, 0] += grads[b, oc, i, j]
                        dX_pad[b, :, h0:h0+k, w0:w0+k] += self.W[oc] * grads[b, oc, i, j]

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        # unpad
        if p > 0:
            return dX_pad[:, :, p:-p, p:-p]
        return dX_pad

    def clear_grad(self):
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}

# -----------------------------------------------------------------------------
# ReLU Activation
# -----------------------------------------------------------------------------
class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def forward(self, X):
        self.input = X
        return np.where(X < 0, 0, X)

    def backward(self, grad):
        return np.where(self.input < 0, 0, grad)

# -----------------------------------------------------------------------------
# Loss: Softmax + Cross-Entropy
# -----------------------------------------------------------------------------
class MultiCrossEntropyLoss(Layer):
    """Softmax + Cross-Entropy Loss"""
    def __init__(self, model=None) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.grads = None

    def __call__(self, logits, labels):
        # override base __call__ to accept both predicts and labels
        return self.forward(logits, labels)

    def forward(self, logits, labels):
        batch = logits.shape[0]
        if self.has_softmax:
            z = logits - np.max(logits, axis=1, keepdims=True)
            ez = np.exp(z)
            probs = ez / np.sum(ez, axis=1, keepdims=True)
        else:
            probs = logits
        self.probs, self.labels = probs, labels
        loss = -np.mean(np.log(probs[np.arange(batch), labels] + 1e-12))
        dlogits = probs.copy()
        dlogits[np.arange(batch), labels] -= 1
        dlogits /= batch
        self.grads = dlogits
        return loss

    def backward(self):
        if self.model is None:
            return self.grads
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

    """Softmax + Cross-Entropy Loss"""
    def __init__(self, model=None) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.grads = None

    def forward(self, logits, labels):
        batch = logits.shape[0]
        if self.has_softmax:
            z = logits - np.max(logits, axis=1, keepdims=True)
            ez = np.exp(z)
            probs = ez / np.sum(ez, axis=1, keepdims=True)
        else:
            probs = logits
        self.probs, self.labels = probs, labels
        loss = -np.mean(np.log(probs[np.arange(batch), labels] + 1e-12))
        dlogits = probs.copy()
        dlogits[np.arange(batch), labels] -= 1
        dlogits /= batch
        self.grads = dlogits
        return loss

    def backward(self):
        if self.model is None:
            return self.grads
        self.model.backward(self.grads)
# -----------------------------------------------------------------------------
# Regularizers
# -----------------------------------------------------------------------------
class L2Regularization(Layer):
    def __init__(self, layers, lambda_val=1e-8) -> None:
        super().__init__()
        self.layers = [l for l in layers if hasattr(l, 'W')]
        self.lambda_val = lambda_val
        self.optimizable = False

    def forward(self):
        return 0.5 * self.lambda_val * sum(np.sum(l.W**2) for l in self.layers)

    def backward(self):
        for l in self.layers:
            l.grads['W'] += self.lambda_val * l.W

class L1Regularization(Layer):
    def __init__(self, layers, lambda_val=1e-8) -> None:
        super().__init__()
        self.layers = [l for l in layers if hasattr(l, 'W')]
        self.lambda_val = lambda_val
        self.optimizable = False

    def forward(self):
        return self.lambda_val * sum(np.sum(np.abs(l.W)) for l in self.layers)

    def backward(self):
        for l in self.layers:
            l.grads['W'] += self.lambda_val * np.sign(l.W)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def softmax(X):
    z = X - np.max(X, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

class Reshape(Layer):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape
        self.input_shape = None
        self.optimizable = False

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape((X.shape[0],) + self.shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------
class Model_MLP(Layer):
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.layers = []
        self.size_list = size_list
        self.act_func = act_func
        if size_list and act_func:
            for i in range(len(size_list)-1):
                lin = Linear(size_list[i], size_list[i+1])
                if lambda_list:
                    lin.weight_decay = True
                    lin.weight_decay_lambda = lambda_list[i]
                self.layers.append(lin)
                if i < len(size_list)-2:
                    if act_func=='ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise NotImplementedError("Only ReLU supported")

    def forward(self, X):
        out = X
        for l in self.layers:
            out = l(out)
        return out

    def backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)
        return grad

    def load_model(self, path):
        data = pickle.load(open(path,'rb'))
        sizes, act, *plist = data
        self.__init__(sizes, act, [p['lambda'] for p in plist])
        idx=0
        for l in self.layers:
            if hasattr(l,'W'):
                p = plist[idx]; idx+=1
                l.W, l.b = p['W'], p['b']
                l.params['W'], l.params['b'] = l.W, l.b
                l.weight_decay, l.weight_decay_lambda = p['weight_decay'], p['lambda']

    def save_model(self, path):
        plist=[{'W':l.params['W'], 'b':l.params['b'],
                'weight_decay':l.weight_decay, 'lambda':l.weight_decay_lambda}
               for l in self.layers if hasattr(l,'W')]
        pickle.dump([self.size_list, self.act_func, *plist], open(path,'wb'))

class Model_CNN(Layer):
    def __init__(self,
                 conv_configs=None,
                 fc_size_list=None,
                 act_func='ReLU',
                 lambda_list_conv=None,
                 lambda_list_fc=None,
                 input_shape=None):
        super().__init__()
        self.layers=[]
        self.input_shape = input_shape
        if input_shape:
            self.layers.append(Reshape(input_shape))
        if conv_configs:
            for i,(ic,oc,k,s,p) in enumerate(conv_configs):
                c = conv2D(ic,oc,k,s,p)
                if lambda_list_conv:
                    c.weight_decay, c.weight_decay_lambda = True, lambda_list_conv[i]
                self.layers += [c, ReLU()]
        if fc_size_list:
            self.layers.append(Flatten())
            for i in range(len(fc_size_list)-1):
                l = Linear(fc_size_list[i],fc_size_list[i+1])
                if lambda_list_fc:
                    l.weight_decay, l.weight_decay_lambda = True, lambda_list_fc[i]
                self.layers.append(l)
                if i < len(fc_size_list)-2:
                    self.layers.append(ReLU())

    def forward(self, X):
        out = X
        for l in self.layers:
            out = l(out)
        return out

    def backward(self, grad):
        for l in reversed(self.layers):
            grad = l.backward(grad)
        return grad

    def load_model(self, path):
        data = pickle.load(open(path,'rb'))
        conv_c, fc_c, act, *plist = data
        self.__init__(conv_c, fc_c, act,
                      [p['lambda'] for p in plist[:len(conv_c)]],
                      [p['lambda'] for p in plist[len(conv_c):]],
                      self.input_shape)
        idx=0
        # now set weights in all Linear and conv2D layers in order
        for l in self.layers:
            if hasattr(l,'W'):
                p = plist[idx]; idx+=1
                l.W, l.b = p['W'], p['b']
                l.params['W'], l.params['b'] = l.W, l.b
                l.weight_decay, l.weight_decay_lambda = p['weight_decay'], p['lambda']

    def save_model(self, path):
        plist=[]
        for l in self.layers:
            if hasattr(l,'W'):
                plist.append({
                    'W':l.params['W'], 'b':l.params['b'],
                    'weight_decay':l.weight_decay, 'lambda':l.weight_decay_lambda
                })
        data = [self.conv_configs, self.fc_size_list, self.act_func, *plist]
        pickle.dump(data, open(path,'wb'))
