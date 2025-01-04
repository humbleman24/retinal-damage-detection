from OCT_loader import oct_loader
from base_resnet import oct_resnet
from base_vit import oct_vit

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt

import datetime
import os



class train_controller:
    def __init__(self, model_weight = None, optimizer_state = None):
        self.batch_size = 32
        self.lr = 5e-3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 1


        self.train_data = oct_loader(data_type="train")
        self.val_data = oct_loader(data_type="val")
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size = len(self.val_data), shuffle = True)
        self.model = oct_vit()
        if model_weight is not None:
            self.model.load_state_dict(torch.load(model_weight))
            print("Loaded the previous model result")
        self.model = self.model.to(self.device)

        
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if optimizer_state is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_state))
            print("Loaded the previous optimizer state")
            

    def run(self):
        train_loss = []
        val_loss = []
        for epoch in range(1, self.epochs + 1):
            print("{} Epoch {} start".format(datetime.datetime.now(), epoch))
            i = 0
            train_loss_10 = 0
            for images, labels in self.train_loader:
                i += 1
                images, labels = images.to(self.device), labels.to(self.device)

                output = self.model(images)

                loss = self.loss_func(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_10 += loss.item()
                if i % 10 == 0:
                    train_loss.append(train_loss_10/10)
                    print(f"{datetime.datetime.now()} Epoch {epoch}, Training Loss: {train_loss[-1]} ")
                    train_loss_10 = 0
            print("{} Epoch {} start Validation".format(datetime.datetime.now(), epoch))  
            j = 0
            for images, labels in self.val_loader:
                j += 1
                with torch.no_grad():  
                    images, labels = images.to(self.device), labels.to(self.device)

                    output = self.model(images)

                    loss = self.loss_func(output, labels)

                    val_loss += [loss.item() for n in range(i // 10)]
                    
                    print(f"{datetime.datetime.now()} Epoch {epoch}, Validation Loss: {val_loss[-1]} ")
    
        plt.figure(figsize=(10, 5))
        plt.plot(range(1,len(train_loss) + 1), train_loss, label='Training Loss')
        plt.plot(range(1,len(val_loss) + 1), val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.savefig("loss_plot_early_stop2.png")
        plt.show()


                


output_dir = "model_safer"

model_save_path = os.path.join(output_dir, "base_vit1.pth")
optimizer_save_path = os.path.join(output_dir, "base_vit_optim1.pth")


# t = train_controller(model_weight=model_save_path, optimizer_state=optimizer_save_path)
t = train_controller()
t.run()


# model_save_path = os.path.join(output_dir, "base_resnet_early_stop2.pth")
# optimizer_save_path = os.path.join(output_dir, "base_resnet_optim_early_stop2.pth")



torch.save(t.model.state_dict(), model_save_path)
torch.save(t.optimizer.state_dict(), optimizer_save_path)



