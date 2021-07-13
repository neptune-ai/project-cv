import hashlib

import neptune.new as neptune
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split
from torchviz import make_dot


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Net(nn.Module):
    def __init__(self, fc_out_features):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, fc_out_features)
        self.fc2 = nn.Linear(fc_out_features, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


PARAMS = {"batch_size": 128,
          "fc_out_features": 64,
          "lr": 0.009,
          "momentum": 0.95,
          "n_epochs": 3}

# Initialize Neptune
run = neptune.init(project="common/project-cv",
                   tags=["pytorch", "CIFAR-10"])

# Log parameters
run["model/params"] = PARAMS

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

un_normalize_img = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

ts = torchvision.datasets.CIFAR10(root="./data", train=True,
                                  download=True, transform=transform)
train_subsets = random_split(ts, [len(ts)-100, 100], generator=torch.Generator().manual_seed(8652))
train_set = train_subsets[0]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=PARAMS["batch_size"],
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root="./data", train=False,
                                        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=PARAMS["batch_size"],
                                          shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# Log data version
run["data/train/version"] = hashlib.md5(np.ascontiguousarray(train_set.dataset.data[train_set.indices])).hexdigest()
run["data/test/version"] = hashlib.md5(np.ascontiguousarray(test_set.data)).hexdigest()

run["data/train/size"] = len(train_set)
run["data/test/size"] = len(test_set)

# Log class names
run["data/classes"] = classes

net = Net(PARAMS["fc_out_features"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=PARAMS["lr"], momentum=PARAMS["momentum"])

for epoch in range(PARAMS["n_epochs"]):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Log batch loss
        run["training/metrics/batch/loss"].log(loss)

        y_true = labels.cpu().detach().numpy()
        y_pred = outputs.argmax(axis=1).cpu().detach().numpy()

        # Log batch accuracy
        run["training/metrics/batch/accuracy"].log(accuracy_score(y_true, y_pred))

        loss.backward()
        optimizer.step()

        # Log image predictions
        if i == len(train_loader)-382:
            for image, label, prediction in zip(inputs, labels, outputs):
                img = image.detach().cpu()
                img_np = un_normalize_img(img).permute(1, 2, 0).numpy()

                pred_label_idx = int(torch.argmax(F.softmax(prediction, dim=0)).numpy())

                name = "pred: {}".format(classes[pred_label_idx])
                desc_target = "target: {}".format(classes[label])
                desc_classes = "\n".join(["class {}: {}".format(classes[i], pred)
                                         for i, pred in enumerate(F.softmax(prediction, dim=0))])
                description = "{} \n{}".format(desc_target, desc_classes)
                run["training/preds/epoch_{}".format(epoch)].log(
                    neptune.types.File.as_image(img_np),
                    name=name,
                    description=description
                )

# Log model weights
torch.save(net.state_dict(), "cifar_net.pth")
run["model/dict"].upload("cifar_net.pth")

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Log test accuracy
run["training/test_accuracy"] = correct / total

# Log sample batch
data = iter(test_loader).next()
for image, label in zip(data[0], data[1]):
    image = image / 2 + 0.5
    run["data/sample/class-{}-({})".format(label, classes[label])].\
        log(neptune.types.File.as_image(np.transpose(image.numpy(), (1, 2, 0))))

# Log model visualization
y = net(data[0])
model_vis = make_dot(y.mean(), params=dict(net.named_parameters()))
model_vis.format = "png"
model_vis.render("model_vis")
run["model/visualization"] = neptune.types.File("model_vis.png")

run.wait()
