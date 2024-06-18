import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

from torch import optim
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from metric import get_iou_score
from networks.modeling import deeplabv3plus_mobilenet

from dataset import CarSegmentation, ToTensor, Augmenter, Normalizer

batch_size = 6
dataset = CarSegmentation(image_dir=r'E:\AI\dataset5',
                          seg_dir=r'E:\AI\dataset5g',
                          transform=transforms.Compose([Augmenter(), Normalizer(), ToTensor()]))
train_loader = DataLoader(dataset, batch_size, shuffle=True)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = deeplabv3plus_mobilenet(num_classes=2).to(device)

criterion = nn.BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.85)
#checkpoint = r"E:\AI\save_dir2\checkpoints\train_220_1.pth"

#model.load_state_dict(torch.load(checkpoint))
save_dir = r'E:\AI\save_dir2'
writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs'), flush_secs=30)

model.train()
for epoch in range(0, 1):
    losses = []
    ious = []
    for iter, sample in enumerate(train_loader):
        image, masks = sample['image'], sample['masks']
        image, masks = image.to(device), masks.to(device)

        optimizer.zero_grad()

        predict = model(image)
        loss = criterion(predict, masks)

        loss.backward()
        optimizer.step()
        #iou = get_iou_score(predict, masks).mean()
        print(epoch, iter, loss.item(), scheduler.get_last_lr())
        plt.matshow(image[0][0])
        plt.show()
        losses.append(loss.item())
        #ious.append()
    scheduler.step()
    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f'train_{epoch}' + '.pth'))

    writer.add_scalar(tag='losses',
                      scalar_value=np.mean(losses),
                      global_step=epoch)

writer.close()
