import os
import time

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import RNet
from utils import imshow


class TrainModel(object):

    def __init__(self, pretrained=False, model_path=None, *args, **kwargs):
        self.model_save = './model'
        self.data_url = './data/102_data/'
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(10),  # 随机角度旋转
                transforms.RandomHorizontalFlip(),  # 概率水平翻转
                transforms.RandomGrayscale(p=.3),  # 以p的概率将图像随机转换为灰度 p=.35
                transforms.Resize(256),
                transforms.CenterCrop(225),
                transforms.ToTensor(),  # 转换为张量
                transforms.RandomErasing(),  # 随机选择图像中的矩形区域并删除其像素
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(225),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.data_url, x),
                self.data_transforms[x]
            ) for x in ['train', 'val']
        }
        data_loader_train = DataLoader(
            self.image_datasets['train'], batch_size=64,
            shuffle=True, num_workers=16, pin_memory=True
        )
        data_loader_val = DataLoader(
            self.image_datasets['val'], batch_size=64,
            shuffle=True, num_workers=16, pin_memory=True
        )

        self.dataloaders = {
            'train': data_loader_train,
            'val': data_loader_val,
        }

        self.net = RNet(pretrained=pretrained)
        if self.model_path:
            print('PyTorch Load state ...')
            self.net.load_state_dict(torch.load(self.model_path))
            print('PyTorch Load state Model OK!')

        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

        self.optimizer = optim.Adam(self.net.fc.parameters(), lr=0.005, weight_decay=0.0001)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)  # step_size=6, .5

        if not os.path.exists(self.model_save):
            os.makedirs(os.path.join(self.model_save, 'best'))

    def train(self, criterion, optimizer, scheduler, num_epochs=5000):
        """
        训练模型
        :param model:
        :param criterion:
        :param optimizer:
        :param scheduler:
        :param num_epochs:
        :return:
        """
        t = time.time()
        model_best_acc = 0.0

        model_best_name = 'best_acc_{:.4f}_loss_{:.4f}_epo_{}'
        dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                criterion_loss = 0.0
                criterion_corrects = 0

                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        for _nu in range(3):
                            pass
                        outputs = self.net(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    criterion_loss += loss.item() * inputs.size(0)
                    criterion_corrects += torch.sum(preds == labels.data)

                epochs_loss = criterion_loss / dataset_sizes[phase]
                epochs_acc = criterion_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    scheduler.step()  # epochs_loss

                """
                Acc： 准确率
                """
                print('{} Loss: {:.8f} Acc: {:.8f}'.format(
                    phase, epochs_loss, epochs_acc)
                )

                # 保留最优模型
                if phase == 'val' and epochs_acc > model_best_acc:
                    model_best_acc = epochs_acc
                    _best_name = model_best_name.format(model_best_acc, epochs_loss, epoch)
                    self.save(self.net, self.model_save, _best_name)

            if epoch % 10 == 0:
                self.save(
                    self.net, self.model_save,
                    'epoch_{}_acc_{:.8f}_loss_{:.8f}'.format(epoch, epochs_acc, epochs_loss)
                )

        t = time.time() - t
        print('Train complete in {:.0f}m {:.0f}s'.format(t // 60, t % 60))
        print('Best val Acc: {:8f}'.format(model_best_acc))

        return self.net

    def test(self, criterion, data='val'):
        """
        训练模型
        :param model:
        :param criterion:
        :param optimizer:
        :param scheduler:
        :param num_epochs:
        :return:
        """

        dataset_sizes = {x: len(self.image_datasets[x]) for x in [data]}
        self.net.eval()

        with torch.no_grad():
            criterion_loss = 0.0
            criterion_corrects = 0

            for inputs, labels in self.dataloaders[data]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                criterion_loss += loss.item() * inputs.size(0)
                criterion_corrects += torch.sum(preds == labels.data)

            epochs_loss = criterion_loss / dataset_sizes[data]
            epochs_acc = criterion_corrects.double() / dataset_sizes[data]

            print('{} Loss: {:.8f} Acc: {:.8f}'.format(
                data, epochs_loss, epochs_acc)
            )

        return

    def save(self, net, path, name):
        """
        保存模型参数
        :param model:
        :param path:
        :param name:
        :return:
        """
        print('Torch Model Save ...')
        torch.save(net.state_dict(), os.path.join(path, '{}.pt'.format(name)))
        print('Model Save Path: {}/{}'.format(path, name))

    def __call__(self, num_epochs=5000, *args, **kwargs):
        """
        入口方法
        :param args:
        :param kwargs:
        :return:
        """
        # self.test(self.net)
        net = self.train(
            self.criterion, self.optimizer,
            self.exp_lr_scheduler, num_epochs
        )
        # self.test(self.criterion)

    pass


if __name__ == '__main__':
    path = os.path.join('model/best/best_4990_94.pt')
    model_path = None

    if os.path.exists(path):
        model_path = path

    train = TrainModel(pretrained=True, model_path=model_path)
    # train()
    train.test(train.criterion)
