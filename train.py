import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
import json
import argparse



def main():
    
    # Start with CPU
    device = torch.device("cpu")

    # Requested GPU
    if input_args.gpu:
        device = torch.device("cuda:0")    

    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu") 
    
    print_every = 40

    #Arguments
    input= getargs()

    data_dir= input.data_dir
    save_dir= input.save_dir
    arch= input.arch
    learning_rate= input.learning_rate
    hidden_layers= input.hidden_layers
    epochs= input.epochs
    gpu= input.gpu
    dropout= input.dropout

    #Pulls in and transforms data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    trainloader, validloader, testloader, class_to_idx = process_data(train_dir, valid_dir, test_dir)

    #Defince input and output size
    output_size= len(class_to_idx)

    #Model run
    model= Classifier(arch, dropout)
    model.class_to_idx = class_to_idx

    criterion= nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(), lr=learning_rate)

    #Training the model                
    model_train(model, epochs, trainloader, criterion, optimizer, device)
    save_checkpoint(arch, save_dir, input_size, output_size, hidden_units, dropout, epochs, learning_rate, optimizer)

def getargs():

    #Inputing the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='flowers')
    parser.add_argument('--save_dir', default='.')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--hidden_layers', default=[1024, 256])
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5, help='Determines probability rate for dropouts')

    return parser.parse_args()


def process_data(data_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms= transforms.Compose([ transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms= transforms.Compose([ transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                                         

    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_data, batch_size=50)

    testloader= torch.utils.data.DataLoader(test_data, batch_size=50)

    return trainloader, validloader, testloader, train_data.class_to_idx

#Model
def Classifier(arch='vgg16', dropout=dropout, input_size=input_size, output_size=output_size, hidden_layers=hidden_layers):

    if arch=='vgg16':
        model = models.vgg16(pretrained = True)
        input_size= 25088
    elif arch== 'densenet121':
        model=models.densenet121(pretrained=True)
        input_size= 1024
    elif arch== 'alexnet':
        model= models.alexnet(pretrained=True)
        input_size=9216
    else:
        print('Model not recognised')
    
    #Freezing parameters of pretrained model
    for param in model.parameters():
        param.requires_grad= False
    
    new_classifier= nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('fc1', nn.Linear(input_size, hidden_layers[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
        ('relu2', nn.ReLU()),
        ('output', nn.Linear(hidden_layers[1], output_size)),
        ('softmax', nn.LogSoftmax(dim = 1))
    ]))
        
    
    model.classifier= new_classifier
    
    return model


#Training function
def model_train(device, model=model, epochs=epochs, trainloader= trainloader, criterion=criterion, optimizer= optimizer):

    model.to(device)    
    steps=0
    print_every=50

    for epoch in range(epochs):
        running_loss=0
        
        #Model training
        for images, labels in trainloader:
            steps+=1
            
            images, labels= images.to(device), labels.to(device)
            optimizer.zero_grad()

            logps= model.forward(images)
            loss= criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            
            #Model evaluation
            if steps % print_every==0:
                valid_loss=0
                accuracy=0
                model.eval()
                
                for vimages, vlabels in validloader:
                    vimages, vlabels= vimages.to(device), vlabels.to(device)
                    
                    with torch.no_grad():
                        vlogps= model.forward(vimages)
                        batch_loss= criterion(vlogps, vlabels)
                        valid_loss +=batch_loss.item()

                        #Accuracy
                        ps= torch.exp(vlogps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals= top_class== vlabels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print( f'Epoch {epoch+1}/{epochs}...'
                      f'Train loss: {running_loss/print_every:.3f}..'
                      f'Valid loss: {valid_loss/len(validloader):.3f}..'
                      f'Test accuracy: {accuracy/len(validloader):.3f}')
                running_loss=0
                model.train()



#Accuracy checking function
def test_run(device, testloader=testloader):
    
    correct=0
    total=0
    model.to(device)
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels= images.to(device), labels.to(device)
            lgps= model(images)
            #Accuracy calcualtion
            _, predicted = torch.max(lgps.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print('Accuracy of test set: {}%'.format((correct / total) * 100))


#Saving the checkpoint
def save_checkpoint(device):
    model.class_to_idx = train_data.class_to_idx


    checkpoint={'arch': arch,
                'input_size': input_size,
                'output_size': output_size,
                'hidden_layers': hidden_layers,
                'dropout': dropout,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'optimizer': optimizer,
                'classifier': classifier,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}


    torch.save(checkpoint, save_dir+'checkpoint.pth')


