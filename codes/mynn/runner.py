import numpy as np
import os
from tqdm import tqdm
from math import ceil

class RunnerM():
    """
    Train, evaluate, save & load a model.
    """
    def __init__(self, model, optimizer, metric, loss_fn,
                 batch_size=32, scheduler=None):
        self.model      = model
        self.optimizer  = optimizer
        self.loss_fn    = loss_fn
        self.metric     = metric
        self.scheduler  = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores   = []
        self.train_loss   = []
        self.dev_loss     = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters  = kwargs.get("log_iters", 100)
        save_dir   = kwargs.get("save_dir", "best_model")

        X_train, y_train = train_set
        X_dev,   y_dev   = dev_set
        batch_size        = self.batch_size

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_score = -np.inf

        for epoch in range(num_epochs):
            # shuffle
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

            num_batches = ceil(len(X_train) / batch_size)
            for iteration in range(num_batches):
                start = iteration * batch_size
                end   = start + batch_size
                x_batch = X_train[start:end]
                y_batch = y_train[start:end]
                if len(x_batch) == 0:
                    break

                # forward
                logits    = self.model(x_batch)
                trn_loss  = self.loss_fn(logits, y_batch)
                trn_acc   = self.metric(logits, y_batch)

                # record
                self.train_loss.append(trn_loss)
                self.train_scores.append(trn_acc)

                # backward & update
                self.loss_fn.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                # evaluate on dev
                dev_acc, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_acc)
                self.dev_loss.append(dev_loss)

                # logging
                if iteration % log_iters == 0:
                    print(f"Epoch {epoch}, Iter {iteration}/{num_batches}")
                    print(f"  [Train] loss={trn_loss:.4f}, acc={trn_acc:.4f}")
                    print(f"  [ Dev ] loss={dev_loss:.4f}, acc={dev_acc:.4f}")

            # save best
            if dev_acc > best_score:
                best_score = dev_acc
                best_path  = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(best_path)
                print(f"New best dev acc: {best_score:.4f} (model saved)")

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits   = self.model(X)
        loss     = self.loss_fn(logits, y)
        acc      = self.metric(logits, y)
        return acc, loss

    def save_model(self, save_path):
        """
        Delegates to the model's save_model method.
        """
        self.model.save_model(save_path)

    def load_model(self, load_path):
        """
        Delegates to the model's load_model method.
        """
        self.model.load_model(load_path)
