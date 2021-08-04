import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models

from torch.autograd import Variable

from utils_2016310526 import *

import numpy as np

root_dataset_dirpath = './VOCdevkit/VOC2012/'
learning_rate = 0.001
num_epochs = 50
batch_size = 4
net = resnet50()

print('load pre-trained model')
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()
dd = net.state_dict()
for k in new_state_dict.keys():
    if k in dd.keys() and not k.startswith('fc'):
        dd[k] = new_state_dict[k]
net.load_state_dict(dd)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


criterion = Loss_yolobased(7, 2, 5, 0.5, device=device)

net.to(device)
net.train()

# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

import os
Annotations = os.path.join(root_dataset_dirpath, 'Annotations/')

xml_files = os.listdir(Annotations)

train_file = open('2016310526_VOC_TRAIN_DATA.txt', 'w')
valid_file = open('2016310526_VOC_VALID_DATA.txt', 'w')
test_file = open('2016310526_VOC_TEST_DATA.txt', 'w')

print("parsing Annotation data...")
count = 0
for xml_file in xml_files:

    image_path = xml_file.split('.')[0] + '.jpg'
    results = parse_rec(Annotations + xml_file)
    if len(results) == 0:
        print("{} files cannot parse... continue...".format(xml_file))
        continue

    if count % 8 == 0:
        valid_file.write(image_path)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            valid_file.write(
                ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(
                    class_name))
        valid_file.write('\n')
        count += 1
        continue

    if count % 9 == 0:
        test_file.write(image_path)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            test_file.write(
                ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(
                    class_name))
        test_file.write('\n')
        count += 1
        continue
    else:
        train_file.write(image_path)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            train_file.write(
                ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(
                    class_name))
        train_file.write('\n')

    count += 1

train_file.close()
valid_file.close()
test_file.close()

print("len of train={}, valid={}, test={}".format(sum(1 for line in open('2016310526_VOC_TRAIN_DATA.txt')), sum(1 for line in open('2016310526_VOC_VALID_DATA.txt')), sum(1 for line in open('2016310526_VOC_TEST_DATA.txt'))))


train_dataset = VOCDATASET(root=os.path.join(root_dataset_dirpath, 'JPEGImages/'), input_file='2016310526_VOC_TRAIN_DATA.txt', train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = VOCDATASET(root=os.path.join(root_dataset_dirpath, 'JPEGImages/'), input_file='2016310526_VOC_VALID_DATA.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

scaler = torch.cuda.amp.GradScaler()

best_test_loss = np.inf

from tqdm import tqdm

for epoch in range(num_epochs):
    net.train()
    loop = tqdm(train_loader, leave=True)

    if epoch == 30 or epoch == 40:
        learning_rate *= 0.1

    total_loss = 0.

    for i, (images, target) in enumerate(loop):
        with torch.no_grad():
            images = Variable(images)
            target = Variable(target)

        images, target = images.to(device), target.to(device)

        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.item()

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(EPOCH='[{}/{}]'.format(epoch+1, num_epochs), current_loss='{0:.4f}'.format(loss.item()), total_loss='{0:.4f}'.format(total_loss/(i + 1)))

    # validation
    validation_loss = 0.0
    net.eval()
    test_loop = tqdm(test_loader, leave=True)
    for i, (images, target) in enumerate(test_loop):
        with torch.no_grad():
            images = Variable(images)
            target = Variable(target)

        images, target = images.to(device), target.to(device)

        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.item()

        test_loop.set_postfix(loss='{0:.4f}'.format(validation_loss/len(test_loader)))
    validation_loss /= len(test_loader)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), '2016310526_BEST.pth')
