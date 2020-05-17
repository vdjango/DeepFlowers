import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models import RNet
from utils import imshow


class TestModel(object):

    def __init__(self, pretrained=False, model_path=None, *args, **kwargs):
        self.model_save = './model'
        self.data_url = './data/102_data/'
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.data_url, x),
                self.data_transforms[x]
            ) for x in ['test']
        }

        data_loader_test = DataLoader(
            self.image_datasets['test'],
            shuffle=True, num_workers=4
        )

        self.dataloaders = {
            'test': data_loader_test
        }
        self.out = []

        self.net = RNet(pretrained=pretrained)
        if self.model_path:
            print('PyTorch Load state ...')
            self.net.load_state_dict(torch.load(self.model_path))
            print('PyTorch Load state Model OK!')

        self.net.to(self.device)

    def test(self, net, datasets='test'):
        """
        测试模型
        :param model:
        :param datasets:
        :return:
        """

        net.eval()
        class_names = self.image_datasets['test'].classes

        with torch.no_grad():
            for inputs, labels in self.dataloaders[datasets]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in range(inputs.size()[0]):
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

    def __call__(self, *args, **kwargs):
        """
        入口方法
        :param args:
        :param kwargs:
        :return:
        """
        self.test(self.net)


if __name__ == '__main__':
    path = os.path.join('model/best/best_4990_94.pt')
    model_path = None

    if os.path.exists(path):
        model_path = path

    test = TestModel(pretrained=True, model_path=model_path)
    # test.save_csv()
    test()
    test.save_csv()

