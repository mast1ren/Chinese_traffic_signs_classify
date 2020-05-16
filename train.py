import torch
from torch import nn
from dataload import trafficDataset
from torch.utils.data import DataLoader
from resnet import ResNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EPOCH = 20
LEARNING_RATE = 0.001

full_data = trafficDataset(label_file_path='./Train/label_train.txt', train=True)
train_size = int(0.9 * len(full_data))
vali_size = len(full_data) - train_size
train_data, vali_data = torch.utils.data.random_split(full_data, [train_size, vali_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
vali_loader = DataLoader(vali_data, batch_size=16, shuffle=True)
loss_func = nn.CrossEntropyLoss()


def train():
    model = ResNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, amsgrad=False, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)

    total_step = len(train_loader)

    for epoch in range(EPOCH):
        torch.cuda.empty_cache()
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.7f}'.format(epoch + 1, EPOCH, step + 1, total_step,
                                                                         loss.item()))
        scheduler.step()

        correct_vali = 0
        total_vali = 0
        for images_vali, target in vali_loader:
            images_vali = images_vali.to(device)
            target = target.to(device)
            with torch.no_grad():
                pred = model(images_vali)
                maxk = max((1, 5))
                target_resize = target.view(-1, 1)
                _, predicted = pred.topk(maxk, 1, True, True)
                loss = loss_func(pred, target)
                correct_vali += torch.eq(predicted, target_resize).sum().item()
                total_vali += target.size(0)
        print('Epoch {} Accuracy {:.4f} %, loss {:.4f}'.format(epoch + 1, 100 * correct_vali / total_vali, loss.item()))

    torch.save(model.state_dict(), 'model.pt')
