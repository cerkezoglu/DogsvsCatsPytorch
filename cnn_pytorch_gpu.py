import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import zipfile
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Extract the zip file to the directory
# local_zip = 'kagglecatsanddogs_3367a.zip'
# zip_ref = zipfile.ZipFile(local_zip,'r')
# zip_ref.extractall()
# zip_ref.close()

torch.device("cpu")
Rebuild_data = True


class DogsVSCats():
    IMG_SIZE = 50
    base_dir = os.path.join('', 'PetImages')

    cats_dir = os.path.join(base_dir, "Cat")
    dogs_dir = os.path.join(base_dir, "Dog")

    labels = {cats_dir:0, dogs_dir:1}
    data = []
    cat_count = 0
    dog_count = 0

    def make_data(self):
        for label in self.labels:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.data.append([np.array(img), np.eye(2)[self.labels[label]]])

                    if label == self.cats_dir:
                        self.cat_count += 1
                    if label == self.dogs_dir:
                        self.dog_count += 1
                except Exception as e:
                    # print(str(e))
                    pass
        print("COUNT DONE")
        np.random.shuffle(self.data)
        np.save("data.npy", self.data)

        print("Cats: ", self.cat_count)
        print("Dogs: ", self.dog_count)

        for label in self.labels:
            print(label)


if Rebuild_data:
    dogsvscats= DogsVSCats()
    dogsvscats.make_data()
    Rebuild_data = False



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self,x):

        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim= 1)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



net = Net().to(device)

data = np.load("data.npy", allow_pickle=True)
print(len(data))


optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()

x = torch.Tensor([i[0] for i in data]).view(-1,50,50)
x = x/255.0
y = torch.Tensor([i[1] for i in data])


VAL_PCT = 0.1

val_size = int(len(x)*VAL_PCT)

train_x = x[val_size:]
train_y = y[val_size:]

test_x = x[:val_size]
test_y = y[:val_size]

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 3


    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_x), BATCH_SIZE,)):

            batch_x = train_x[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y = train_y[i:i+BATCH_SIZE]


            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = loss_function(outputs, batch_y)
            loss.backward()

            optimizer.step()
    print(loss)


def test(net):
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_x))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_x[i].view(-1,1,50,50))
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
        print("Accuracy: ", round(correct/total, 3))

def fwd_pass(x, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(x)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss



def test(size = 32):

    random_start = np.random.randint(len(test_x)-size)
    x, y = test_x[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(x.view(-1,1,50,50).to(device), y.to(device), train=False)
    return val_acc, val_loss


val_acc, val_loss = test(size=32)

print(val_acc,val_loss)

import time

MODEL_NAME = f"model-{int(time.time())}"

net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()
print(MODEL_NAME)

def train_new():
    BATCH_SIZE = 100
    EPOCHS = 6
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0,len(train_x),BATCH_SIZE)):
                batch_x = train_x[i:i+BATCH_SIZE].view(-1,1,50,50).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)

                acc,loss = fwd_pass(batch_x, batch_y, train = True)
                if i % 50 == 0:
                    val_acc, val_loss = test(size= 100)
                    f.write(f"{MODEL_NAME}, {round(time.time(), 3)}, {round(float(acc), 2)}, {round(float(loss), 4)}, {round(float(val_acc), 2)}, {round(float(val_loss), 4)}\n")

train_new()
