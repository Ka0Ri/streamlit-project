from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


imsize = 512 if torch.cuda.is_available() else 128 

normalization = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

unnormalization = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)),
    transforms.ToPILImage()])


def image_loader(image_name):
    """
    function : load an image
    input: path name
    output: tensor
    """
    image = Image.open(image_name)
    image = normalization(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    """
    function : show an image with title
    input: tensor, title
    output: none
    """
    plt.figure(figsize=(8,8))
    image = tensor.cpu().clone()  
    image = image.squeeze(0)     
    image = unnormalization(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 


plt.ion()

style_img = image_loader("anime.jpg")
content_img = image_loader("theo.jpg")


imshow(style_img, title='Style Image')

imshow(content_img, title='Content Image')


class ContentLoss(nn.Module):
    """
    function: calculate contenloss
    input: content layer of target content image
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        """
        function: calculate contenloss
        input: content layer of generated image
        """
        loss = F.mse_loss(input, self.target)
        return loss
    
def gram_matrix(input):
    """
    function: calculate Gram matrix
    input: a style layer of an image
    output: gram matrix
    """
    a, b, c, d = input.size()
    features = input.view(a * b, c * d) 

    G = torch.mm(features, features.t())  # compute the gram product

    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    """
    function: calculate style loss
    input: style layers of target style image
    """
    def __init__(self, targets_feature):
        super(StyleLoss, self).__init__()
        self.target = [gram_matrix(target).detach() for target in targets_feature] 

    def forward(self, input):
        """
        function: calculate style loss
        input: style layers of generated image
        output: 
        """
        loss = 0
        for i in range(len(input)):
            G = gram_matrix(input[i])
            loss += F.mse_loss(G, self.target[i])
        return loss
    
class Reconstructionloss(nn.Module):
    """
    function: calculate total loss
    input: style layers and content of target images
    alpha: content weight, beta: style weight 
    """
    def __init__(self, style_target, content_target, alpha, beta):
        super(Reconstructionloss, self).__init__()
        self.contentloss = ContentLoss(content_target)
        self.styleloss = StyleLoss(style_target)
        self.alpha = alpha
        self.beta = beta

    def forward(self, style_input, content_input):
        
        return self.alpha * self.contentloss(content_input) + self.beta * self.styleloss(style_input)


tensor_unnormalization = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)),
    ])

class style_transfer():
    
    def __init__(self, style_target, content_target):
        """
        input: style_target: style image
               content_target: content image
        """
        # reused network
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        self.cnn = copy.deepcopy(cnn)
        print(self.cnn)
        self.content_layers_default = ['28']
        self.style_layers_default = ['5', '10', '19', '28', '34']
        
#         self.content_layers_default = ['layer3']
#         self.style_layers_default = ['layer1', 'layer2', 'layer3', 'layer4']
        
        content_layers = self.extract_content(content_target)
        style_layers = self.extract_style(style_target)
        
        self._loss = Reconstructionloss(style_layers, content_layers, alpha=1, beta=10e6)
        
    def extract_style(self, x):
        """
        function: extract style layer
        input: target image
        """
        outputs = []
        for name, module in self.cnn._modules.items():
            x = module(x)
            if name in self.style_layers_default:
                outputs += [x]
        
        return outputs
        
    def extract_content(self, x):
        """
        function: extract content layer
        input: target image
        """
        outputs = []
        for name, module in self.cnn._modules.items():
            x = module(x)
            if name in self.content_layers_default:
                outputs += [x]
                
        return outputs[0]
    
    def blending(self, content_img, num_steps=300):
        
        input_img = content_img.clone()
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        
        output = input_img.clone()
        output = tensor_unnormalization(output.squeeze(0))
        save_image(output, "t/img[0].png")
        
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image

                optimizer.zero_grad()
                
                styles = self.extract_style(input_img)
                content = self.extract_content(input_img)
                
                loss = self._loss(styles, content)
                
                
                loss.backward()

                run[0] += 1
                if run[0] % 5 == 0:
                    print("run {}:".format(run))
                    print('Loss : {:4f} '.format(loss.item()))
                    output = input_img.clone()
                    output = tensor_unnormalization(output.squeeze(0))
                    save_image(output, "t/img{}.png".format(run))
                    
                return loss
                   
            optimizer.step(closure)
        
        return input_img


st = style_transfer(style_img, content_img)

output = st.blending(content_img)