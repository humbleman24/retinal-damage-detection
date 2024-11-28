from OCT_loader import oct_loader
from base_resnet import oct_resnet

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import datetime
import os



class train_controller:
    def __init__(self, model_weight = None, optimizer_state = None):
        self.batch_size = 32
        self.lr = 1e-4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 19


        self.train_data = oct_loader(data_type="train")
        self.test_data = oct_loader(data_type="test")
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size = self.batch_size, shuffle = True)
        if model_weight is not None:
            self.model = oct_resnet()
            self.model.load_state_dict(torch.load(model_weight))
            self.model = self.model.to(self.device)
        
        self.loss_func = nn.CrossEntropyLoss()
        if optimizer_state is not None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(torch.load(optimizer_state))

    def run(self):
        for epoch in range(1, self.epochs + 1):
            print("{} Epoch {} start".format(datetime.datetime.now(), epoch))
            i = 0
            for images, labels in self.train_loader:
                i += 1
                images, labels = images.to(self.device), labels.to(self.device)

                output = self.model(images)

                loss = self.loss_func(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i in [1, 10, 20, 50]:
                    print(f"{datetime.datetime.now()} Epoch {epoch}, Training Loss: {loss.item()} ")
                elif i % 500 == 0:
                    print(f"{datetime.datetime.now()} Epoch {epoch}, Training Loss: {loss.item()} ")
                


output_dir = "model_safer"

model_save_path = os.path.join(output_dir, "base_resnet.pth")
optimizer_save_path = os.path.join(output_dir, "base_resnet_optim.pth")


t = train_controller(model_weight=model_save_path, optimizer_state=optimizer_save_path)
t.run()


torch.save(t.model.state_dict(), model_save_path)
torch.save(t.optimizer.state_dict(), optimizer_save_path)



