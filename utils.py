import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import joblib
import numpy as np

class MyModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
        self.model = ""
        self.binarizer = ""
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.num_classes = ""
        self.load_binarizer()
        self.load_model()

    def load_binarizer(self):
        mlb = joblib.load("./model/mlb.pkl")
        self.num_classes = len(mlb.classes_)
        self.binarizer = mlb

    def load_model(self):
        model = MyModel(n_classes=self.num_classes)
        model.load_state_dict(torch.load(f="./model/parameters.pth", map_location=torch.device('cpu')))
        self.model = model

    def predict(self, image, num_of_tags=5):
        image_transformed = self.transform(image)
        batch_image = torch.unsqueeze(image_transformed, 0)
        self.model.eval()

        # Inference
        output = self.model(batch_image).detach().numpy()
        preds = output[0]
        top_n_tags =  np.sort(preds)[::-1][min(num_of_tags - 1, len(preds)-1)]
        preds[preds < top_n_tags] = 0
        preds[preds >= top_n_tags] = 1
        tags = self.binarizer.inverse_transform(np.array([preds]))[0][:]
        return tags