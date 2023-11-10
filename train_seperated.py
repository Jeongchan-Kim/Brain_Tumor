import torch
from tqdm import tqdm

def train(train_dl, epoch, optimizer, model, criterion):
    model.train()
    train_losses = []
    for batch in tqdm(train_dl):
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Epoch: {} train_loss: {:.4f}".format(epoch, torch.stack(train_losses).mean().item()))