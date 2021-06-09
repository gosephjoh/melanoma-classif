from typing import Type
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd

from load_images import load_images
import models


class Arguments:
    def __init__(self, dropout: float, model_type: Type[models.BasicNet],
                 num_epochs: int, lr: float, wd: float, batch_size: int):
        # Setup the CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Model arguments
        self.dropout: float = dropout
        self.model_type: Type[nn.Module] = model_type

        # Training arguments
        self.num_epochs: int = num_epochs
        self.lr: float = lr
        self.wd: float = wd
        self.batch_size: int = batch_size


def train_and_eval(train_loader: DataLoader, test_loader: DataLoader, args: Arguments) -> pd.DataFrame:
    model = args.model_type(args)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize DataFrame to log training scores for later visualization
    eval_dict = {'epoch': [],
                 'train_acc': [], 'test_acc': [],
                 'train_loss': [], 'test_loss': []
                }

    for e in range(args.num_epochs):
        # Training
        model.train()
        train_targets, train_preds = [], []
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_preds += predictions.detach().cpu().numpy().tolist()
            train_targets += labels.detach().cpu().numpy().tolist()

        train_acc = accuracy_score(train_preds, train_targets)

        # Evaluation
        model.eval()
        test_targets, test_preds = [], []
        test_loss = 0.0

        for i, data in enumerate(test_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)

            test_preds += predictions.detach().cpu().numpy().tolist()
            test_targets += labels.detach().cpu().numpy().tolist()

            test_loss += loss_fn(outputs, labels).item()

        test_acc = accuracy_score(test_preds, test_targets)

        # Logging the epoch and scores

        eval_dict['epoch'].append(e)
        eval_dict['train_acc'].append(train_acc)
        eval_dict['test_acc'].append(test_acc)
        eval_dict['train_loss'].append(train_loss)
        eval_dict['test_loss'].append(test_loss)
        print(f'Epoch {e}, Train={train_acc:.4f}, Test={test_acc:.4f}')

    return pd.DataFrame.from_dict(eval_dict)


if __name__ == "__main__":
    train_loader, test_loader = load_images("./data", 32, 0.8, 0.5)
    args = Arguments(dropout=0.5, model_type=models.AlexNet, num_epochs=40, lr=0.0001, wd=0, batch_size=32)
    train_and_eval(train_loader, test_loader, args)
