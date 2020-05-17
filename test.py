import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import RNet
from utils import imshow


class TestModel(object):

    def __init__(self, pretrained=False, *args, **kwargs):
        self.model_save = './model'
        self.data_url = './data/54_data/'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
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
            self.image_datasets['train'],
            shuffle=True, num_workers=4
        )
        data_loader_val = DataLoader(
            self.image_datasets['val'],
            shuffle=True, num_workers=4
        )

        self.dataloaders = {
            'train': data_loader_train,
            'val': data_loader_val
        }
        self.out = []

        self.net = RNet(pretrained=pretrained)
        self.net.to(self.device)

    def test(self, net, datasets='test'):
        """
        测试模型
        :param model:
        :param datasets:
        :return:
        """
<<<<<<< HEAD

=======
>>>>>>> parent of 9a54e7c... 基于resnet152模型的迁移学习-102花朵分类
        net.eval()
        class_names = self.image_datasets['train'].classes

        with torch.no_grad():
            for inputs, labels in self.dataloaders[datasets]:
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in range(inputs.size()[0]):
<<<<<<< HEAD
                    # class_predicted = class_names[predicted[i]]  # DELETE
                    self.out.append([labels.cpu().item(), predicted[i].cpu().item()])
                    # print('predicted:', predicted[i].cpu().item())
                    # imshow(inputs.cpu().data[i], title='predicted: {}'.format(class_predicted))

    def save_csv(self):
        import csv

        with open('a.csv', 'w', newline='') as f:
            c = csv.writer(f)
            for _cs in self.out:
                c.writerow(_cs)
        pass
=======
                    class_predicted = class_names[predicted[i]]
                    print('predicted:', class_predicted)
                    imshow(inputs.cpu().data[i], title='predicted: {}'.format(class_predicted))
>>>>>>> parent of 9a54e7c... 基于resnet152模型的迁移学习-102花朵分类

    def __call__(self, *args, **kwargs):
        """
        入口方法
        :param args:
        :param kwargs:
        :return:
        """
        self.test(self.net)


if __name__ == '__main__':
<<<<<<< HEAD
    path = os.path.join('model/best/best_4990_94.pt')
    model_path = None

    if os.path.exists(path):
        model_path = path

    test = TestModel(pretrained=True, model_path=model_path)
    # test.save_csv()
=======
    test = TestModel(pretrained=True)
>>>>>>> parent of 9a54e7c... 基于resnet152模型的迁移学习-102花朵分类
    test()
    test.save_csv()

