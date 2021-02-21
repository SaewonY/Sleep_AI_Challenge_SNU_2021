# from conf import *

import torch 
import torch.nn as nn
import torchvision.models as models
import timm

sigmoid = torch.nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
swish_layer = Swish_module()

def build_model(args, device):
        
    if args.model == 'base': 
        model = CNN_Base().to(device)
    else:
        model_list = list()
        model_list.append(nn.Conv2d(1, 3, 1))
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
        model_list.append(model)
        model = nn.Sequential(*model_list)
        if args.use_meta:
            model = timm_models(args)

    if device: 
        model = model.to(device)
    
    return model


class timm_models(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_list = list()
        model_list.append(nn.Conv2d(1, 3, 1))
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=512)
        model_list.append(model)
        model = nn.Sequential(*model_list)
        self.dropout = nn.Dropout(0.5)                
        self.dropouts = nn.ModuleList([
                    nn.Dropout(0.5) for _ in range(5)])
        self.model = model
        str_model = nn.Sequential(
            nn.Linear(4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            Swish_module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_module(),
        )
        self.str_model = str_model
        self.output_layer = nn.Linear(512 + 128, args.num_classes)

    def extract(self, x):
        x=self.model(x)
        return x

    def forward(self, img, str_feature):
        #img_feat = self.model(img)
        img_feat = self.extract(img).squeeze(-1).squeeze(-1)
        str_feat = self.str_model(str_feature)

        feat = torch.cat([img_feat, str_feat], dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i==0:
                output = self.output_layer(dropout(feat))
            else:
                output += self.output_layer(dropout(feat))
        else:
            output /= len(self.dropouts)
        #output = self.output_layer(self.dropout(feat))
        return output

class Resnet50(nn.Module):
    def __init__(self, num_classes, dropout=False, pretrained=False):
        super().__init__()
        model = models.resnet50(pretrained=pretrained)
        model = list(model.children())[:-1]
        if dropout:
            model.append(nn.Dropout(0.2))
        model.append(nn.Conv2d(2048, num_classes, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()  

        self.cnn_layer = nn.Sequential(            
            nn.Conv2d(3,6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6,12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(12,15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential( 
            nn.Linear(735, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), 
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
