from .op import *
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

    def __init__(self, conv_configs=None, fc_size_list=None, act_func='ReLU', lambda_list_conv=None, lambda_list_fc=None, input_shape=None):
        super().__init__()
        self.layers=[]
        if input_shape: self.layers.append(Reshape(input_shape))
        if conv_configs:
            for i,(ic,oc,k,s,p) in enumerate(conv_configs):
                c=conv2D(ic,oc,k,s,p)
                if lambda_list_conv: c.weight_decay=c.weight_decay_lambda=lambda_list_conv[i]
                self.layers+=[c,ReLU()]
        if fc_size_list:
            self.layers.append(Flatten())
            for i in range(len(fc_size_list)-1):
                l=Linear(fc_size_list[i],fc_size_list[i+1])
                if lambda_list_fc: l.weight_decay=l.weight_decay_lambda=lambda_list_fc[i]
                self.layers+=[l]+([ReLU()] if i<len(fc_size_list)-2 else [])
                
    def __call__(self, X):
        """Make the model instance callable: forwards to .forward()"""
        return self.forward(X)

    def forward(self,X):
        out=X
        for l in self.layers: out=l(out)
        return out

    def backward(self,grad):
        for l in reversed(self.layers): grad=l.backward(grad)
        return grad

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            params = pickle.load(f)
        # first three entries: conv_configs, fc_size_list, act_func
        self.conv_configs, self.fc_size_list, self.act_func = params[:3]
        self.layers = []
        idx = 3
        # reshape
        if self.input_shape is not None:
            self.layers.append(Reshape(self.input_shape))
        # load convs
        for cfg in self.conv_configs:
            in_c, out_c, k, s, p = cfg
            conv = conv2D(in_c, out_c, k, stride=s, padding=p)
            p_dict = params[idx]; idx+=1
            conv.W = p_dict['W']; conv.b = p_dict['b']
            conv.params['W'], conv.params['b'] = conv.W, conv.b
            conv.weight_decay = p_dict['weight_decay']
            conv.weight_decay_lambda = p_dict['lambda']
            self.layers.append(conv)
            self.layers.append(ReLU())
        # flatten + FC
        self.layers.append(Flatten())
        for _ in range(len(self.fc_size_list)-1):
            lin = Linear(self.fc_size_list[_], self.fc_size_list[_+1])
            p_dict = params[idx]; idx+=1
            lin.W = p_dict['W']; lin.b = p_dict['b']
            lin.params['W'], lin.params['b'] = lin.W, lin.b
            lin.weight_decay = p_dict['weight_decay']
            lin.weight_decay_lambda = p_dict['lambda']
            self.layers.append(lin)
            if _ < len(self.fc_size_list)-2:
                self.layers.append(ReLU())

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