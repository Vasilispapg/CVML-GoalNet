import torch.nn as nn
import torch.nn.functional as F
import torch


def cnn0():
    return Cnn0()

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class Cnn0(nn.Module):

    def __init__(self):
        super(Cnn0, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32,64,5,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = SeparableConv2d(64,128,5,2,1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(128,256,3,2,1)
        self.bn4 = nn.BatchNorm2d(256)

    def features(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def forward(self, input):
        x = self.features(input)
        return x

class audio_visual_model(nn.Module):
    def __init__(self):
        super(audio_visual_model, self).__init__()
        # Visual Branch (Xception)
        # self.visual_model = xception(num_classes=1000)
        self.visual_model = cnn0()
        self.visual_model.fc = nn.Identity()  # Adapt final layer based on Xception architecture
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Audio Branch (Simple CNN for MFCC); the sliding window covers all the Mel coefficients, for some fixed time units totalling <kernel_size>, and slides across the time unit axis; therefore for a given sliding window, the output considers all Mel coefficients for these particular fixed time units and returns one value
        self.audio_model = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, audio_input, visual_input): # audio_input: MFCC features, visual_input: Frames
        # display_tensor_info(tnsr = visual_input, var_name = 'visual_input')
        # AUDIO FEATURES
        audio_features = self.audio_model(audio_input)
        # VISUAL FEATURES
        with torch.no_grad():
            features = self.visual_model(visual_input)  # Extract features using Xception
        visual_features = nn.ReLU(inplace=True)(features)
        visual_features = F.adaptive_avg_pool2d(visual_features, (1, 1))
        visual_features = visual_features.view(visual_features.size(0), -1)
        visual_features = nn.LazyLinear(512).to(visual_features.device)(visual_features)

        combined_features = torch.cat((audio_features, visual_features), axis = -1)

        output = 4 * self.fusion(combined_features) + 1
        return output