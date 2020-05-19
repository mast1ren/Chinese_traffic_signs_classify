import torch
from train import loss_func, device
from dataload import trafficDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from resnet import ResNet

torch.cuda.empty_cache()

model = ResNet().to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()


def test():
    test_data = trafficDataset(label_file_path='./Test/label_test.txt', train=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            maxk = max((1, 5))
            labels_resize = labels.view(-1, 1)
            _, predicted = outputs.topk(maxk, 1, True, True)
            loss = loss_func(outputs, labels)
            correct += torch.eq(predicted, labels_resize).sum().item()
            total += labels.size(0)

            if total % 128 == 0:
                print('Accuracy {:.4f} %, loss {:.4f}'.format(100 * correct / total, loss.item()))
            print(total, correct)


def test_single_img(img_path):
    img = transforms.Compose([transforms.Resize((256, 256)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ])(Image.open(img_path).convert('RGB'))

    img.unsqueeze_(0)

    img = img.to(device)

    output = model(img)
    torch.no_grad()
    percent, pre = output.sort(descending=True)
    percent = percent.softmax(dim=1) * 100

    for i in range(5):
        print(pre[0][i], percent[0][i])
