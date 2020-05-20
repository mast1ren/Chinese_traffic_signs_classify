from train import *
from test import *
from resnet import ResNet

# train()

torch.cuda.empty_cache()

model = ResNet().to(device)
model.load_state_dict(torch.load('./model&score/model.pt'))
model.eval()

# test(model)
test_single_img('./Test/000_0006_j.png', model=model)
