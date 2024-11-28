from OCT_loader import oct_loader
from base_resnet import oct_resnet

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import datetime



class train_controller:
    def __init__(self):
        self.batch_size = 32
        self.lr = 1e-4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 20


        self.train_data = oct_loader(data_type="train")
        self.test_data = oct_loader(data_type="test")
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size = self.batch_size, shuffle = True)

        self.model = oct_resnet().to(self.device)
        
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

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
                    print(f"{datetime.datetime.now()} Epoch {epoch + 1}, Training Loss: {loss.item()} ")
                elif i % 5000 == 0:
                    print(f"{datetime.datetime.now()} Epoch {epoch + 1}, Training Loss: {loss.item()} ")
                

t = train_controller()
t.run()


