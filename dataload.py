from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class trafficDataset(Dataset):
    def __init__(self, label_file_path, train=True):
        f = open(label_file_path, 'r')
        self.data = list(map(lambda line: line.strip().split(' '), f))
        self.train = train

    def __getitem__(self, index):
        path, label = self.data[index]
        if self.train:
            img = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.RandomRotation([-30, 30]),
                                      transforms.RandomGrayscale(0.3),
                                      transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                      ])(Image.open(path).convert('RGB'))
        else:
            img = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                      ])(Image.open(path).convert('RGB'))
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.data)
