import numpy as np
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_data = pd.read_csv("/kaggle/input/dfl-bundesliga-data-shootout/train.csv")
train_data["file_path"] = "/kaggle/input/resized-data/" + train_data["video_id"] + ".mp4"
VEDIO_IDS = train_data.video_id.unique()
print(VEDIO_IDS)
print(len(VEDIO_IDS))


vd_time = []
v_ids = []
print(VEDIO_IDS)
for i in VEDIO_IDS:
    #print(i)
    v_d_current = train_data[train_data["video_id"]==i]
    length = len(v_d_current.time)
    print(length)
    time_duration = round(v_d_current.iloc[-1].time)
    #print(time_duration)
    time_duration_v = np.arange(0,time_duration-2,1)
    vd_time.extend(time_duration_v)
    v_ids.extend([i]*len( time_duration_v))
    print(len(vd_time),"      ",len(v_ids))

v_labels = pd.DataFrame({"video_id":v_ids, "time":vd_time})


labels_used = []
paths = []
for i, row in v_labels.iterrows():
    train_data_current = train_data[train_data.video_id == row.video_id]
    train_data_current['time'] = (train_data_current['time'] - row.time).abs()
    closest_timestamp = min(train_data_current["time"])
    #print(closest_timestamp)
    closest_timestamp_index = train_data_current[train_data_current.time == closest_timestamp].index.tolist()
    #print(closest_timestamp_index)
    labels_used.append(train_data_current.event[closest_timestamp_index[0]])
    paths.append(train_data_current.file_path[closest_timestamp_index[0]])


v_labels["event"] = labels_used
v_labels["file_path"] = paths



import os
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class Input_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.time_split = 25   # 25 frames per second, 5 seconds for interval

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        event = self.data.iloc[index].event
        file_path = self.data.iloc[index].file_path
        time = self.data.iloc[index].time
        Label = {"challenge": 0, "throwin": 1, "play": 2, "start": 3, "end": 4}
        label = Label[event]
        a1 = torch.tensor([0, 0, 0 ,0, 0])
        idex = list(Label.keys()).index(str(event))
        a1[idex] = 1
        cap = cv2.VideoCapture(file_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, time * self.time_split+1)
        video_data = np.zeros((5, 128, 128, 3))
        video_data = video_data.transpose(0, 3, 1, 2)
        for i in range(5):
            _, frame = cap.read()
            frame = cv2.resize(frame, (128, 128))
            frame = frame / 255.0
            frame = torch.Tensor(frame)
            frame = frame.permute(2, 0, 1)
            video_data[i] = frame

        return video_data, label,a1


train, val = train_test_split(v_labels, test_size=0.1, random_state=42, stratify = v_labels.event)

batch_size = 1
train_loader = DataLoader(
    Input_Dataset(train),
    batch_size=batch_size,
    shuffle=False,
    num_workers=1
)

val_loader = DataLoader(
     Input_Dataset(val),
     batch_size=batch_size,
     shuffle=False,
     num_workers=1
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input size (1, 128, 128)
            nn.Conv2d(
                in_channels=3,
                out_channels=10,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),  # relu layer
            nn.AvgPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 5, 1, 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 30, 5, 1, 2),
            nn.AvgPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(30, 50, 5, 1, 2),
            # nn.BatchNorm2d(30, momentum=0.1),
            # nn.AvgPool2d(kernel_size=2),
            nn.ReLU(),
        )

    def forward(self, i):
        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # view: equal to  reshape
        x = x.view(i.shape[0], i.shape[1], -1)  # flatten

        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(12800, 100)
        self.fc = nn.Linear(500, 5)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.model_cnn = CNN().to(device)
        self.model_lstm = LSTM().to(device)

    def forward(self, x):
        features = self.model_cnn(x)
        out = self.model_lstm(features)
        return out

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)


# model = CNN_LSTM().to(device)


dataloaders_dict = {"train": train_loader, "val": val_loader}


def train(model, epochs):
    # model.load_weights("/kaggle/input/pre-trained-weight/vgg16_bn-6c64b313.pth")
    # model.load_weights("/kaggle/input/pre-traine-weight/CNN_LSTM_1.pth")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_func = nn.CrossEntropyLoss(reduction='mean').cuda()

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    initial_acc = 0.0
    for epoch in range(epochs):
        model.cuda()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                # print(item[1])
                frames = item[0].cuda().float()
                classes = item[1].cuda().long()
                o_h_l = item[2].cuda().float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(frames)
                    output1 = torch.squeeze(output)
                    s = nn.Softmax(dim=0)
                    output2 = s(output1)
                    output2 = torch.unsqueeze(output2, 0)
                    loss = loss_func(output2, o_h_l)

                    _, preds = torch.max(output2, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # StepLR.step()

                    running_loss += loss.item() * len(output)
                    running_acc += torch.sum(preds == classes.data)

            data_size = len(dataloader.dataset)
            running_loss = running_loss / data_size
            running_acc = running_acc.double() / data_size
            if phase == "train":
                train_loss.append(running_loss)
                train_acc.append(running_acc.cpu())
            else:
                val_loss.append(running_loss)
                val_acc.append(running_acc.cpu())
            print("Epoch:{0} ||  Loss:{1}".format(epoch, format(loss, ".4f")))

        if running_acc > initial_acc:
            torch.save(model.state_dict(), 'CNN_LSTM_12.pth')
            initial_acc = running_acc
    return train_loss, val_loss, train_acc, val_acc
model = CNN_LSTM().to(device)

model.load_weights("/kaggle/input/pretrainedweights/vgg16.pth")
train(model,1)
