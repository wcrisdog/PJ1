from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    Fully-connected layer: Y = X W + b
    """
    def __init__(self, in_dim, out_dim,
                 initialize_method=np.random.normal,
                 weight_decay=False,
                 weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return X.dot(self.W) + self.b

    def backward(self, grad):
        batch_size = self.input.shape[0]
        self.grads['W'] = self.input.T.dot(grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        return grad.dot(self.W.T)

    def clear_grad(self):
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}

class conv2D(Layer):
    """
    2D convolution layer (with padding & stride)
    """
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
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        batch, _, H, W = X.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        H_out = (H + 2*p - k)//s + 1
        W_out = (W + 2*p - k)//s + 1
        X_pad = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
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
        if p > 0:
            return dX_pad[:, :, p:-p, p:-p]
        return dX_pad

    def clear_grad(self):
        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}

    
    def clear_grad(self):
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.grads = None
        self.max_classes = max_classes

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        batch = predicts.shape[0]
        if self.has_softmax:
            # numeric stability
            probs = softmax(predicts)
        else:
            probs = predicts
        self.probs = probs
        self.labels = labels

        # cross-entropy
        correct_logprobs = -np.log(probs[np.arange(batch), labels] + 1e-12)
        loss = np.mean(correct_logprobs)

        dlogits = probs.copy()
        dlogits[np.arange(batch), labels] -= 1
        dlogits /= batch
        self.grads = dlogits

        return loss
    
    def backward(self):
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, layers, lambda_val=1e-8) -> None:
        super().__init__()
        # only layers that have W
        self.layers = [l for l in layers if hasattr(l, 'W')]
        self.lambda_val = lambda_val
        self.optimizable = False

    def forward(self):
        reg_loss = 0.0
        for l in self.layers:
            reg_loss += np.sum(l.W ** 2)
        return 0.5 * self.lambda_val * reg_loss

    def backward(self):
        for l in self.layers:
            # add regularization gradient to weight grads
            if 'W' in l.grads:
                l.grads['W'] += self.lambda_val * l.W

class Dropout(Layer):
    """
    Dropout layer: during training, zero each unit with probability p.
    Scales remaining activations by 1/(1-p) to keep expectation constant.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.mask = None
        self.optimizable = False

    def forward(self, X, training=True):
        if not training or self.p == 0:
            return X
        # create dropout mask
        self.mask = (np.random.rand(*X.shape) > self.p) / (1.0 - self.p)
        return X * self.mask

    def backward(self, grad):
        # only propagate through the surviving units
        return grad * (self.mask if self.mask is not None else 1.0)

       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False
    
    def __call__(self, X):
        return self.forward(X)

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
    
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape((X.shape[0],) + self.shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)