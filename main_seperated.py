import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loader

from preprocessing import random_seed, class_preprocessing, data_preprocessing
from model import ResNet18
from train_seperated import train
from validation import val
from test import test
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu', dest='gpu', type=int, default=0)
parser.add_argument('-seed', '--seed', dest='seed', type=int, default=724)
parser.add_argument('-val_size', '--val_size', dest='val_size', type=float, default=0.2)
parser.add_argument('-m', '--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('-w', '--weight_decay', dest='weight_decay', type=float, default=0.0001)
parser.add_argument('-f', '--factor', dest='factor', type=float, default=0.1)
parser.add_argument('-lr', '--lr', dest='lr', type=float, default=5.5e-5)
parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=8)
parser.add_argument('-p', '--patience', dest='patience', type=int, default=5)
parser.add_argument('-e', '--max_epoch', dest='max_epoch', type=int, default=3)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    random_seed(random_seed=args.seed)
    class_preprocessing()
    data_preprocessing(val_size=args.val_size)
    train_dl, val_dl, test_dl = get_data_loader(batch_size=args.batch_size)
    model = ResNet18().to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.max_epoch

    for epoch in range(epochs):
        train(train_dl, epoch, optimizer, model, criterion)
        val(val_dl, epoch, model, criterion)
    
    test(test_dl, model)