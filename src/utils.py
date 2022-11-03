import torch
from tqdm import tqdm



def train(data_loader, model, criterion, optimizer, device):
    running_accuracy = 0
    running_loss = 0
    
    model.train()
    for X, y in tqdm(data_loader):
        optimizer.zero_grad()

        _input = {}

        y = y.to(device)
        for key, value in X.items():
            _input[key] = value.squeeze().to(device)
        

        # forward pass
        out = model(_input)

        # loss
        loss = criterion(out.squeeze(), y)
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

    
def test(data_loader, model, criterion, device):
    running_accuracy = 0
    running_loss = 0
    
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            _input = {}

            y = y.to(device)
            for key, value in X.items():
                _input[key] = value.squeeze().to(device)
            

            # forward pass
            out = model(_input)

            # loss
            loss = criterion(out.squeeze(), y)
            running_loss += loss.item()

            # accuracy
            preds = torch.round( torch.sigmoid(out.detach()) )
            acc = (y == preds).sum() / len(y)
            running_accuracy += acc.item()

        mean_loss = running_loss / len(data_loader)
        mean_accuracy = running_accuracy / len(data_loader)
        return mean_loss, mean_accuracy



