from op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self,conv_configs=None,fc_size_list=None,act_func='ReLU',lambda_list_conv=None,lambda_list_fc=None):
        super().__init__()
        self.conv_configs = conv_configs
        self.fc_size_list = fc_size_list
        self.act_func = act_func
        self.layers = []
        if conv_configs is not None:
            for idx, cfg in enumerate(conv_configs):
                in_c, out_c, k, s, p = cfg
                conv = conv2D(in_c, out_c, k, stride=s, padding=p)
                if lambda_list_conv:
                    conv.weight_decay = True
                    conv.weight_decay_lambda = lambda_list_conv[idx]
                self.layers.append(conv)
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                else:
                    raise NotImplementedError
        if fc_size_list is not None:
            self.layers.append(Flatten())
            for i in range(len(fc_size_list) - 1):
                lin = Linear(fc_size_list[i], fc_size_list[i+1])
                if lambda_list_fc:
                    lin.weight_decay = True
                    lin.weight_decay_lambda = lambda_list_fc[i]
                self.layers.append(lin)
                if i < len(fc_size_list) - 2:
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise NotImplementedError

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            params = pickle.load(f)
        self.conv_configs, self.fc_size_list, self.act_func = params[0], params[1], params[2]
        self.layers = []
        idx = 3
        for cfg in self.conv_configs:
            in_c, out_c, k, s, p = cfg
            conv = conv2D(in_c, out_c, k, stride=s, padding=p)
            p_dict = params[idx]
            conv.W = p_dict['W']
            conv.b = p_dict['b']
            conv.params['W'], conv.params['b'] = conv.W, conv.b
            conv.weight_decay = p_dict['weight_decay']
            conv.weight_decay_lambda = p_dict['lambda']
            idx += 1
            self.layers.append(conv)
            if self.act_func == 'ReLU':
                self.layers.append(ReLU())
            else:
                raise NotImplementedError
        self.layers.append(Flatten())
        for _ in range(len(self.fc_size_list) - 1):
            lin = Linear(self.fc_size_list[_], self.fc_size_list[_+1])
            p_dict = params[idx]
            lin.W = p_dict['W']
            lin.b = p_dict['b']
            lin.params['W'], lin.params['b'] = lin.W, lin.b
            lin.weight_decay = p_dict['weight_decay']
            lin.weight_decay_lambda = p_dict['lambda']
            idx += 1
            self.layers.append(lin)
            if _ < len(self.fc_size_list) - 2:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                else:
                    raise NotImplementedError

    def save_model(self, save_path):
        param_list = [self.conv_configs, self.fc_size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

