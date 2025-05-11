# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'./codes/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./codes/dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx_2.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

cnn_model = nn.models.Model_CNN(
    conv_configs=[(1, 32, 3, 1, 1),
                  (32, 64, 3, 1, 1)],
    fc_size_list=[64*28*28, 600, 10],
    act_func='ReLU',
    lambda_list_conv=[1e-4, 1e-4],
    lambda_list_fc=[1e-4, 1e-4],
    input_shape=(1, 28, 28)
)


linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
optimizer = nn.optimizer.SGD(init_lr=0.01, model=cnn_model)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5)
loss_fn   = nn.op.MultiCrossEntropyLoss(model=cnn_model)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=100, save_dir=r'./codes/saved_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()