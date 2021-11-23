import argparse
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import select_device
from models import CustomCNN
from utils.utils import plot_accuracy_and_loss


def test(model, test_loader, device):
    predictions_list = []
    labels_list = []
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels_list.append(labels)
            outputs = model(images)
            predictions = torch.max(outputs, 1)[1].to(device)
            predictions_list.append(predictions)
            correct += (predictions == labels).sum()
            total += len(labels)
        
        accuracy = correct * 100 / total
    print(f"Test accuracy: {accuracy}")


def train(opt):
    device = select_device(opt.device)

    # prepare dataset
    # transforms
    train_transformers = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))    
    ])

    val_transformers = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=train_transformers)
    val_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=val_transformers)  

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=opt.batch_size)
    val_loader = DataLoader(val_set, batch_size=2*opt.batch_size)

    model = CustomCNN()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    learning_rate = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 5
    count = 0
    # Lists for visualization of loss and accuracy 
    loss_list = []
    iteration_list = []
    accuracy_list = []

    val_loss_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    patience = 0
    last_accuracy = 0.0
    best_model = None

    for epoch in range(opt.num_epochs):
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass 
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            #Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1

        # Testing the model
        with torch.no_grad():
            total = 0
            correct = 0
            val_loss = None
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                outputs = model(images)
                val_loss = loss_fn(outputs, labels)

                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)
            
            accuracy = correct * 100 / total
            if last_accuracy == None:
                last_accuracy = accuracy
                best_model = copy.deepcopy(model)
            elif accuracy <= last_accuracy:
                patience += 1
            else:
                # reset patience if accuracy improves
                patience = 0
                best_model = copy.deepcopy(model)
                last_accuracy = accuracy
            
            loss_list.append(loss.data.cpu().numpy())
            val_loss_list.append(val_loss.data.cpu().numpy())

            iteration_list.append(count)
            accuracy_list.append(accuracy.cpu())
                
            print("Epoch: {}, Loss: {}, Val Loss {}, Accuracy: {}%".format(epoch, loss.data, val_loss.data, accuracy))
        
        if patience >= opt.patience:
            print(f"Early stopping score hasnt improved for 5 epochs")
            break
    
    # save model
    torch.save(best_model.state_dict(), opt.model_path)
    print(f"Saved model successfully!")

    # plot results on train and validation sets
    plot_accuracy_and_loss(accuracy_list, loss_list, val_loss_list, "models/results")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/weights/fashion_mnist_cnn.pt', help='initial weights path')
    parser.add_argument('--train-csv', type=str, default='data/fashion_mnist/fashion-mnist_train.csv', help='train.csv path')
    parser.add_argument('--test-csv', type=str, default='data/fashion_mnist/fashion-mnist_test.csv', help='test.csv path')
    parser.add_argument('--batch-size', type=int, default=256, help='total batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--num-epochs', type=int, default=50, help='total epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print(opt)
    train(opt)