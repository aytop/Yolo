from torchvision.models import AlexNet
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Alex(AlexNet):
    def __init__(self):
        super(Alex, self).__init__()
        self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Conv2d(256, 15, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256, 9, 9)
        x = self.classifier(x)
        return x
