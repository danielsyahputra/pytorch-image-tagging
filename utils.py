import torch
from torch import nn
from torchvision import transforms, models
import joblib

class MyModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigmoid = nn.Sigmoid()
        self.feature_layer = self.base_model._modules.get("avgpool")

    def forward(self, x):
        return self.sigmoid(self.base_model(x))

    def get_image_vector(self, img, model, device):
        image = img.unsqueeze(0).to(device)
        embedding = torch.zeros(1, 512, 1, 1)
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.feature_layer.register_forward_hook(copyData)
        model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]

class MyPredictor():
    def __init__(self) -> None:
        pass

    def load_model(self):
        pass