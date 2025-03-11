from OCT_loader import oct_loader
from base_resnet import oct_resnet
from base_vit import oct_vit
from squeeze_vit import SqueezeViT

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt

import datetime
import os
from torch.utils.tensorboard import SummaryWriter



class train_controller:
    def __init__(self, model_weight = None, optimizer_state = None):
        self.batch_size = 32
        self.lr = 5e-3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 10

        # load the data into dataloader
        self.train_data = oct_loader(data_type="train")
        self.val_data = oct_loader(data_type="val")
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size = len(self.val_data), shuffle = True)

        # model configuration
        self.model = SqueezeViT()
        if model_weight is not None:       # if the weight is provided, load the weight
            self.model.load_state_dict(torch.load(model_weight))
            print("Loaded the previous model result")
        self.model = self.model.to(self.device)

        # loss and optimizer
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if optimizer_state is not None:    # if the optimizer state is provided, load the state
            self.optimizer.load_state_dict(torch.load(optimizer_state))
            print("Loaded the previous optimizer state")
            

    def run(self):
        # initialize tensorboard writer
        writer  = SummaryWriter()

        # add graph to tensorboard
        sample_images, _ = next(iter(self.train_loader))
        writer.add_graph(self.model, sample_images.to(self.device))

        for epoch in range(1, self.epochs + 1):
            print("{} Epoch {} start".format(datetime.datetime.now(), epoch))
            for idx, (images, labels) in enumerate(self.train_loader, 1):
                images, labels = images.to(self.device), labels.to(self.device)     # remember to device!

                output = self.model(images)

                loss = self.loss_func(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                writer.add_scalar("Loss/train", loss.item(), epoch * len(self.train_loader) + idx)

                if idx % 100 == 0:
                    print(f"{datetime.datetime.now()} Epoch {epoch}, Steps: {idx}, Training Loss: {loss.item()} ")

            # validation
            print("{} Epoch {} start Validation".format(datetime.datetime.now(), epoch))  
            for idx, (images, labels) in enumerate(self.val_loader, 1):
                with torch.no_grad():  
                    images, labels = images.to(self.device), labels.to(self.device)

                    output = self.model(images)

                    loss = self.loss_func(output, labels)

                    writer.add_scalar('Loss/val', loss.item(), epoch * len(self.val_loader) + idx)
                    
                    print(f"{datetime.datetime.now()} Epoch {epoch}, , Steps: {idx}, Validation Loss: {loss.item()} ")
    
        writer.close()

                


output_dir = "model_safer"

model_save_path = os.path.join(output_dir, "svit.pth")
optimizer_save_path = os.path.join(output_dir, "svit.pth")


# t = train_controller(model_weight=model_save_path, optimizer_state=optimizer_save_path)
t = train_controller()
t.run()

# model_save_path = os.path.join(output_dir, "base_resnet_early_stop2.pth")
# optimizer_save_path = os.path.join(output_dir, "base_resnet_optim_early_stop2.pth")


torch.save(t.model.state_dict(), model_save_path)
torch.save(t.optimizer.state_dict(), optimizer_save_path)



