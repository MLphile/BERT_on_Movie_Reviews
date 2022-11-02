import torch
from tqdm import tqdm
import re


def train(data_loader, model, criterion, optimizer, device):
    running_accuracy = 0
    running_loss = 0
    
    model.train()
    for X, y in tqdm(data_loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # forward pass
        out = model(X)

        # loss
        loss = criterion(out, y)
        running_loss += loss.item()

        # backward pass      
        loss.backward()
        
        # update paramaters
        optimizer.step()

        # accuracy
        preds = torch.round( torch.sigmoid(out.detach()) )
        acc = (y == preds).sum() / len(y)
        running_accuracy += acc.item()
    # # mean loss (all batches losses divided by the total number of batches)
    # train_losses.append(running_loss / len(trainloader))
    
    # # mean accuracies
    # train_accuracies.append(running_accuracy / len(trainloader))
    mean_loss = running_loss / len(data_loader)
    mean_accuracy = running_accuracy / len(data_loader)

    return mean_loss, mean_accuracy

    
def train(data_loader, model, criterion, optimizer, device):
    running_accuracy = 0
    running_loss = 0
    
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            # forward pass
            out = model(X)

            # loss
            loss = criterion(out, y)
            running_loss += loss.item()

            # accuracy
            preds = torch.round( torch.sigmoid(out.detach()) )
            acc = (y == preds).sum() / len(y)
            running_accuracy += acc.item()

        mean_loss = running_loss / len(data_loader)
        mean_accuracy = running_accuracy / len(data_loader)
        return mean_loss, mean_accuracy



def clean(text):
    # remove weird spaces
    text =  " ".join(text.split())
    # remove html tags
    text = re.sub(r'<.*?>', '', text)
    return text