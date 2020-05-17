import torch
from torch import nn
from dataload import trafficDataset
from torch.utils.data import DataLoader
from resnet import ResNet
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EPOCH = 50
LEARNING_RATE = 0.001
# EPOCH = 50
# LEARNING_RATE = 0.01

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
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    total_step = len(train_loader)

    plot_x = []
    plot_acc = []
    plot_recall = []
    plot_precision = []
    plot_f1 = []
    plot_x_loss = []
    plot_loss = []

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
            if step % 50 == 0:
                plot_x_loss.append(step / 235 + epoch)
                plot_loss.append(loss.item())
        scheduler.step()

        true_target = torch.FloatTensor()
        pre_target = torch.FloatTensor()
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

                temp = predicted.narrow(1, 0, 1)
                temp = torch.squeeze(temp, 1)
                temp = temp.cpu()
                temp = temp.view(-1).float()
                pre_target = torch.cat((pre_target, temp), 0)
                temp = target.cpu()
                temp = temp.view(-1).float()
                true_target = torch.cat((true_target, temp), 0)

        score_precision, score_recall, score_f1, _ = precision_recall_fscore_support(true_target, pre_target,
                                                                                     average='macro')
        plot_x.append(epoch + 1)
        plot_acc.append(correct_vali / total_vali)
        plot_f1.append(score_f1)
        plot_precision.append(score_precision)
        plot_recall.append(score_recall)

        print(
            'Epoch {} Accuracy {:.4f} %, loss {:.4f}\nf1-score {:.4f} recall {:.4f} precision {:.4f}'.format(epoch + 1,
                                                                                                             100 * correct_vali / total_vali,
                                                                                                             loss.item(),
                                                                                                             score_f1,
                                                                                                             score_recall,
                                                                                                             score_precision))
    plotScore(plot_x, plot_acc, plot_recall, plot_precision, plot_f1, plot_x_loss, plot_loss)

    torch.save(model.state_dict(), 'model.pt')


def plotScore(plot_x, plot_acc, plot_recall, plot_precision, plot_f1, plot_x_loss, plot_loss):
    plt.figure(figsize=(10, 10), dpi=100)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.5)
    plt.subplot(grid[0, 0])
    plt.plot(plot_x, plot_acc, 'o-b')
    plt.title('accurary', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accurary', fontsize=14)

    plt.subplot(grid[0, 1])
    plt.plot(plot_x, plot_recall, 'o-b')
    plt.title('recall', fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('recall', fontsize=14)

    plt.subplot(grid[1, 0])
    plt.plot(plot_x, plot_precision, 'o-b')
    plt.title('precision', fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('precision', fontsize=14)

    plt.subplot(grid[1, 1])
    plt.plot(plot_x, plot_f1, 'o-b')
    plt.ylim(0, 1)
    plt.title('f1', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('f1', fontsize=14)

    plt.subplot(grid[2, 0:2])
    plt.plot(plot_x_loss, plot_loss, 'o-b')
    plt.title('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)

    plt.savefig(fname='score.svg', format='svg')

    plt.show()
