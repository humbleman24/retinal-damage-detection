from typing import Optional
import os
import glob
from PIL import Image

import torchvision
from torchvision import transforms
from torchvision.transforms.transforms import Compose
import torch
from torch.utils.data import Dataset

Data_path = "../archive/OCT2017_"


class oct_loader(Dataset):
    def __init__(
            self,
            data_type: str = "train",
            transformers: Optional[Compose] = None,
        ) -> None:
        # receive the data type and get the data path
        if data_type == "train" or data_type == "test" or data_type == "val":
            data_path = os.path.join(Data_path, data_type)
        else:
            raise ValueError("The type of dataset should be among 'train', 'test' and 'val', But get {}".format(data_type))
        
        if transformers == None:
            # standard transformation for the image
            transformers = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.transformers = transformers
        
        self.image_paths = []
        self.image_labels = []
        cate_label_mapping = {
            "CNV": 0,
            "DME": 1,
            "DRUSEN": 2,
            "NORMAL": 3
        }
        # loading the image path and label
        for i in os.listdir(data_path):
            label = cate_label_mapping[i]         # get the label based on dir name
            image_path_i = glob.glob(os.path.join(data_path, i) + "/*.jpeg")    # loop over all the images in the dir, return a list
            self.image_paths += image_path_i
            image_label_i = [label] * len(image_path_i)
            self.image_labels += image_label_i

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label = self.image_labels[idx]

        # load image and transfer it only when it is needed
        image = Image.open(img_name).convert("RGB")
        image = self.transformers(image)

        # the order is important, first image, then label
        return image, label


if __name__ == "__main__":
    oct = oct_loader()
    print(len(oct))
