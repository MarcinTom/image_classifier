#Imports 
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
from PIL import Image
import argparse

def load_checkpoint(filepath='checkpoint.pth'):
    checkpoint= torch.load(filepath)
    arch= checkpoint['arch']
    output_size= checkpoint['output_size']
    input_size= checkpoint['input_size']
    learning_rate= checkpoint['learning_rate']
    hidden_layers= checkpoint['hidden_layers']
    epochs= checkpoint['epochs']
    optimizer= checkpoint['optimizer']
    dropout= checkpoint['dropout']
    
    model= Classifier(arch, dropout)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    
    return model
    


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    pict = Image.open(image)
    img= pict.copy()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    np_array = transform(img).float()
    
    return np_array
    

def predict(image_path, model, topk=5):
        
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device)
    
    # Predict the class from an image file
    processed_image = process_image(image_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probs, labels = output.topk(topk)
    probs = np.array(probs.exp().data)[0]
    labels = np.array(labels)[0]
        
    return probs, labels


 def main(image_path, checkpoint, top_k, category_names, gpu):
 	with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

 	model= load_checkpoint('checkpoint.pth')
 	device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")

 	probs, labels = predict(image_path, model, device, top_k)
    
    label_map = {v: k for k, v in model.class_to_idx.items()}
    classes = [cat_to_name[label_map[l]] for l in labels]
    
    return classes, probs

 if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict an image class using classfier',
    )
    
    parser.add_argument('image_path', default='flowers/train/47/image_04985.jpg')
    parser.add_argument('checkpoint', default='checkpoint.pth')
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    input_args = parser.parse_args()

    classes, probs = main(input_args.image_path, input_args.checkpoint, input_args.top_k,
                          input_args.category_names, input_args.gpu)
    
    print([x for x in zip(classes, probs)])

