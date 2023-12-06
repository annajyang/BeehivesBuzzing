import torch
import torch.nn as nn
import torchvision.models as models

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.model = nn.Sequential(     
            nn.Conv2d(330, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),

            nn.Linear(6656, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self,input):
        x = self.model(input)
        return nn.Sigmoid()(x)
        
class BaselineFusionCNN(nn.Module):
    def __init__(self,num_aux):
        super(BaselineFusionCNN, self).__init__()
        self.model = nn.Sequential(     
            nn.Conv2d(330, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),

            nn.Linear(6656, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.Tanh()
        )
        self.out = nn.Sequential(
            nn.Linear(16+num_aux,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self,image,auxvar):
        x = self.model(image)
        x = self.out(torch.cat((x,auxvar),dim=1))
        return nn.Sigmoid()(x)
        
"""
class BaselineColorCNN(nn.Module):
    def __init__(self, input_size=64):
    super(BaselineColorCNN, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(num_classes=365) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.image_features = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):
    image_features = self.image_features(input)
    output = self.upsample(image_features)
    return output


class BaselineColorCNN(nn.Module):
  def __init__(self, input_size=64):
    super(BaselineColorCNN, self).__init__()
    resnet = models.resnet18(num_classes=1) #classes don't matter, last layer not used
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    self.network = nn.Sequential(     
      *list(resnet.children())[0:6],
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )
  def forward(self, input):
    output = self.network(input)
    return output



class ClassifierColorCNN(nn.Module):
  def __init__(self,Q):
    super(ClassifierColorCNN, self).__init__()

    self.color_probs = nn.Sequential(     
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(64),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(128),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(512),

        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(512),

        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        
        nn.Conv2d(256, Q, kernel_size=3, stride=1, padding=1),
    )

    self.upsample = nn.Upsample(scale_factor=4, mode="bilinear",align_corners=False)
    

  def forward(self, input):
    colors = self.color_probs(input)
    output = self.upsample(colors)
    return output


class ClassifierColorUNet(nn.Module):
  def __init__(self,Q=313):
    super(ClassifierColorUNet, self).__init__()

    self.l1 = nn.Sequential(     
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(64))
    
    self.model1short8= nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True))

    self.l2 = nn.Sequential(     
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(128))
    
    self.model2short7= nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True))

    self.l3 = nn.Sequential(     
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(256))
    
    self.model3short6= nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True))

    self.l4 = nn.Sequential(     
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(512))

    self.l5 = nn.Sequential(     
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(512))
    
    self.l6 = nn.Sequential(     
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(512))
    
    self.l7 = nn.Sequential(     
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        nn.BatchNorm2d(256))

    self.l8 = nn.Sequential(     
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
        
    )
    self.out = nn.Conv2d(128, Q, kernel_size=3, stride=1, padding=1)

    self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",align_corners=False)
    

  def forward(self, input):
    out1 = self.l1(input)
    out2 = self.l2(out1) 
    out3 = self.l3(out2) 
    out4 = self.l4(out3) 
    out5 = self.l5(out4) 
    out6 = self.l6(out5)  + self.model3short6(out3)
    out7 = self.l7(out6)  + self.model2short7(out2)
    out8 = self.l8(out7)  + self.model1short8(out1)

    output = self.upsample(self.out(out8))
    return output
"""