
import os
import time
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import copy
import streamlit as st


latest_iteration = st.empty()
bar = st.progress(0)


st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Image Blending by Neural Style Transfer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128 

normalization = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

unnormalization = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225)),
    transforms.ToPILImage()])

def main():

    import streamlit as st

    st.sidebar.title("Settings")
    n_iters = st.sidebar.slider("Number of iteration", 0,  100, 10)

    content_coeff = st.sidebar.slider("Content coefficient", 0.0,  1.0, 1.0)
    content_layer = st.sidebar.selectbox("Content layer",
    ("5", "10", "19", "28", "34"), index=3)

    content_img = load_img("Choose a content image")
    if content_img is not None:
        imshow(content_img)

    style_img = load_img("Choose a style image")
    if style_img is not None:
        imshow(style_img)



    
    st.write("Blended image")
    if  st.button("Blending"):
        imageLocation = st.empty()
        bar = st.progress(0)
        st = style_transfer(style_img, content_img, alpha=content_coeff, cont_layer=content_layer)

        output = st.blending(content_img, num_steps= n_iters)
   

    return


def imshow(tensor, title=None):
    """
    function : show an image with title
    input: tensor, title
    output: none
    """
    image = tensor.cpu().clone()  
    image = image.squeeze(0)     
    image = unnormalization(image)
    st.image(image)


def load_img(s):

    im = None
    content_file = st.file_uploader(s, type=["png", 'jpg', 'jpeg', 'tiff'])
    if content_file is not None:
        im = Image.open(content_file)
        im = normalization(im).unsqueeze(0)
        im = im.to(device, torch.float)
    return im



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
    
    def __init__(self, style_target, content_target, alpha, cont_layer):
        """
        input: style_target: style image
               content_target: content image
        """
        # reused network
        cnn = models.vgg19(pretrained=True).features.to(device).eval()
        self.cnn = copy.deepcopy(cnn)
        self.content_layers_default = [cont_layer]
        self.style_layers_default = ['5', '10', '19', '28', '34']
        
        content_layers = self.extract_content(content_target)
        style_layers = self.extract_style(style_target)
        
        self._loss = Reconstructionloss(style_layers, content_layers, alpha=alpha, beta=10e6)
        
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

    
    # @st.cache 
    def blending(self, content_img, num_steps=10):
        
        input_img = content_img.clone()
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        
        output = input_img.clone()
        
        run = [0]
        latest_iteration = st.empty()
        imageLocation = st.empty()
        bar = st.progress(0)
        while run[0] < (num_steps):
           
            def closure():
                # correct the values of updated input image

                optimizer.zero_grad()
                
                styles = self.extract_style(input_img)
                content = self.extract_content(input_img)
                
                loss = self._loss(styles, content)
                
                
                loss.backward()

                bar.progress(run[0] * 50 // num_steps)
                latest_iteration.text(f'Iteration {run[0]}')
                run[0] += 1
                
                if run[0] % 5 == 0:
                    output = input_img.clone()
                    output = tensor_unnormalization(output.squeeze(0))
                    save_image(output, "t/img.png")
                    imageLocation.image("t/img.png")
                    
                return loss
            
                   
            optimizer.step(closure)
            
        
        return input_img


if __name__ == "__main__":
    main()