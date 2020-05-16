from train import *
from resnet import ResNet

torch.cuda.empty_cache()

train()

test_data = trafficDataset(label_file_path='./Test/label_test.txt', train=False)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

model = ResNet().to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()

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
