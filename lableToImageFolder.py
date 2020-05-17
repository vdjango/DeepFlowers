import os
import shutil
import sys

url = '/Users/vdjango/AI/54_data'  # 102花朵数据集
save_url = '/Users/vdjango/web/ai/DeepFlowers/data/102_data/'  # 处理后的数据集存放路径

image_dataset_path = {
    'train_data': os.path.join(save_url, 'train'),
    'val_data': os.path.join(save_url, 'val'),
    'test_data': os.path.join(save_url, 'test'),
    'train': os.path.join(url, 'train'),
    'test': os.path.join(url, 'test')
}

image_train_path = {}
image_val_path = {}

if not os.path.exists(url):
    print('路径不存在')

if not os.path.exists(image_dataset_path['train_data']):
    os.makedirs(image_dataset_path['train_data'])
    pass

if not os.path.exists(image_dataset_path['val_data']):
    os.makedirs(image_dataset_path['val_data'])

with open(os.path.join(url, 'train.txt')) as file:
    for line in file.readlines():
        s = line.split()
        _name = s[0]
        _class_id = s[1]

        source = os.path.join(image_dataset_path['train'], _name)

        train_target_out = os.path.join(image_dataset_path['train_data'], _class_id)
        val_target_out = os.path.join(image_dataset_path['val_data'], _class_id)

        if not os.path.exists(train_target_out):
            os.makedirs(train_target_out)

        if not os.path.exists(val_target_out):
            os.makedirs(val_target_out)

        if not image_train_path.get(_class_id, None):
            image_train_path[_class_id] = []

        image_train_path[_class_id].append({
            'source': source,
            'train_target': os.path.join(train_target_out, _name),
            'val_target': os.path.join(val_target_out, _name),
        })

    pass

for i in image_train_path:
    _n = int(len(image_train_path[i]) * .2)
    if not image_val_path.get(i, None):
        image_val_path[i] = []

    for val in image_train_path[i][:_n]:
        image_val_path[i].append(val)

    image_train_path[i] = image_train_path[i][_n:]

for i in image_train_path:
    for _target in image_train_path[i]:
        source = _target['source']
        target = _target['train_target']

        if not os.path.exists(target):
            try:
                shutil.copy(source, target)
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())

for i in image_val_path:
    for _target in image_val_path[i]:
        source = _target['source']
        target = _target['val_target']

        if not os.path.exists(target):
            try:
                shutil.copy(source, target)
                # print(source, target)
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())

if os.path.exists(image_dataset_path['test']):
    if not os.path.exists(image_dataset_path['test_data']):
        os.makedirs(image_dataset_path['test_data'])

        for root, dirs, files in os.walk(image_dataset_path['test']):
            for file in files:
                out = os.path.join(image_dataset_path['test_data'], '0')
                if not os.path.exists(out):
                    os.makedirs(out)

                src_file = os.path.join(root, file)
                shutil.copy(src_file, out)
                print(src_file)
