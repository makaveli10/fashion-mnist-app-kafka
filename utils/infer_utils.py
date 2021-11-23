import json
import torch
import numpy as np
from torch.serialization import load
import torchvision.transforms as transforms

from models.classifier import CustomCNN
from utils.utils import output_label


class Classfier:
    def __init__(self, weights, logger) -> None:
        self.logger = logger
        self.model = self.load_model(weights)
        self.val_transformers = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def img_to_array(self, img_stream):
        '''
        Convert Incoming Image to Numpy Array.
        Args:
            img_stream: Incoming Image from Stream.

        Returns:
            np.array
        '''
        try:
            image = np.array(img_stream).astype('float32')
            return image
        except Exception as e:
            self.logger.warning(f"Failed to convert the image to np array {e}")
            return None


    def preprocess(self, img_stream):
        '''Preprocess image.
        '''
        img = self.img_to_array(img_stream)
        image = self.val_transformers(img).unsqueeze(dim=0)
        return image


    def load_model(self, state_dict_path):
        '''Load CNN Model.
        '''
        self.logger.info("loading model ...")
        model = CustomCNN()
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def classify(self, consumedMessage):
        '''Read consumed message and perform classification
        '''
        ids, labels = [], []
        for i in range(len(consumedMessage)):
            x = json.loads(consumedMessage[i])
            for k, v in x.items():
                img = self.preprocess(v)
                outputs = self.model(img)
                predicted = torch.max(outputs, 1)[1]
                label = output_label(predicted)
                ids.append(k)
                labels.append(label)
        return ids, labels