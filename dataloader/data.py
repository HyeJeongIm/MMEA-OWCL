import numpy as np
from collections import OrderedDict
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from .mydataset import MyDataSet
from utils.transforms import ArrayToTensor, DataStack, GroupNormalize, IdentityTransform, ImgStack, ToTorchFormatTensor, GroupScale, GroupCenterCrop
from dataloader.data_class_order import UESTC_MMEA_CLASS_ORDER, DATAEGO_CLASS_ORDER


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iData_TBN(iData):
    use_path = False
    class_order = None

    def __init__(self, model, modality, arch, train_list, test_list):
        self.modality = modality
        self.arch = arch
        self.train_list = train_list
        self.test_list = test_list
        
        self.crop_size = model.backbone.crop_size
        self.scale_size = model.backbone.scale_size
        self.input_mean = model.backbone.input_mean
        self.input_std = model.backbone.input_std
        self.data_length = model.backbone.new_length
        self.train_augmentation = model.backbone.get_augmentation()

        self.train_trsf = {}
        self.test_trsf = {}
        self.normalize = {}

    def download_data(self):
        # normalize
        for m in self.modality:
            if (m != 'RGBDiff'):
                self.normalize[m] = GroupNormalize(self.input_mean[m], self.input_std[m])
            else:
                self.normalize[m] = IdentityTransform()

        # transform
        for m in self.modality:
            if (m != 'Gyro' and m != 'Acce'):
                # Prepare train/val dictionaries containing the transformations
                # (augmentation+normalization)
                # for each modality
                self.train_trsf[m] = transforms.Compose([
                self.train_augmentation[m],
                ImgStack(roll=self.arch == 'BNInception'),
                ToTorchFormatTensor(div=self.arch != 'BNInception'),
                self.normalize[m],
                ])

                self.test_trsf[m] = transforms.Compose([
                    GroupScale(int(self.scale_size[m])),
                    GroupCenterCrop(self.crop_size[m]),
                    ImgStack(roll=self.arch == 'BNInception'),
                    ToTorchFormatTensor(div=self.arch != 'BNInception'),
                    self.normalize[m],
                ])
            else:
                self.train_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                    self.normalize[m],
                ])

                self.test_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                    self.normalize[m],
                ])

        train_set = MyDataSet(self.train_list)
        test_set = MyDataSet(self.test_list)

        self.train_data, self.test_data = np.array(train_set.video_list), np.array(test_set.video_list)
        self.train_targets, self.test_targets = np.array(self._get_targets(train_set)), np.array(self._get_targets(test_set))

    def _get_targets(self, dataset):
        targets = []
        for i in range(len(dataset)):
            targets.append(dataset.video_list[i].label)

        return targets
    

class iData_TSN(iData):
    use_path = False
    class_order = None

    def __init__(self, model, modality, arch, train_list, test_list):
        self.modality = modality
        self.arch = arch
        self.train_list = train_list
        self.test_list = test_list

        self.crop_size = model.backbone.crop_size
        self.scale_size = model.backbone.scale_size
        self.input_mean = model.backbone.input_mean
        self.input_std = model.backbone.input_std
        self.data_length = model.backbone.new_length
        self.train_augmentation = model.backbone.get_augmentation()

        self.train_trsf = {}
        self.test_trsf = {}
        self.normalize = {}

    def download_data(self):

        for m in self.modality:
            if (m != 'Acce' and m != 'Gyro'):
                self.normalize[m] = GroupNormalize(self.input_mean[m], self.input_std[m])

        for m in self.modality:
            if m == 'Acce' or m == 'Gyro':
                self.train_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                ])

                self.test_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                ])

            else:
                self.train_trsf[m] = transforms.Compose([
                    self.train_augmentation[m],
                    ImgStack(roll=self.arch == 'ViT'),
                    ToTorchFormatTensor(),
                    self.normalize[m],
                ])

                self.test_trsf[m] = transforms.Compose([
                    GroupScale(int(self.scale_size[m])),
                    GroupCenterCrop(self.crop_size[m]),
                    ImgStack(roll=self.arch == 'ViT'),
                    ToTorchFormatTensor(),
                    self.normalize[m],
                ])

        train_set = MyDataSet(self.train_list)
        test_set = MyDataSet(self.test_list)

        self.train_data, self.test_data = np.array(train_set.video_list), np.array(test_set.video_list)
        self.train_targets, self.test_targets = np.array(self._get_targets(train_set)), np.array(
            self._get_targets(test_set))

    def _get_targets(self, dataset):
        targets = []
        for i in range(len(dataset)):
            targets.append(dataset.video_list[i].label)

        return targets


# UESTC-MMEA
class iUESTC_MMEA_TBN(iData_TBN):
    def __init__(self, model, modality, arch, train_list, test_list):
        super().__init__(model, modality, arch, train_list, test_list)
        self.class_order = UESTC_MMEA_CLASS_ORDER
        self.input_mean = OrderedDict({
            'RGB': [104, 117, 128],
            'Flow': [128],
            'Acce': [0.0352, 0.3717, -0.7944],
            'Gyro': [78.5445, -2.1253, -6.6940],
        })
        self.input_std = OrderedDict({
            'RGB': [1, 1, 1],
            'Flow': [1],
            'Acce': [0.1836, 0.4058, 0.2219],
            'Gyro': [352.5285, 181.1698, 286.6291],
        })

class iUESTC_MMEA_TSN(iData_TSN):
    def __init__(self, model, modality, arch, train_list, test_list):
        super().__init__(model, modality, arch, train_list, test_list)
        self.class_order = UESTC_MMEA_CLASS_ORDER
        self.input_mean = OrderedDict({
            'RGB':  [.485, .456, .406],
            'Acce': [.485, .456, .406],
            'Gyro': [.485, .456, .406]
        })
        self.input_std = OrderedDict({
            'RGB':  [.229, .224, .225],
            'Acce': [.229, .224, .225],
            'Gyro': [.229, .224, .225]
        })

# DataEgo
class iDataEgo_TBN(iData_TBN):
    def __init__(self, model, modality, arch, train_list, test_list):
        super().__init__(model, modality, arch, train_list, test_list)
        self.class_order = DATAEGO_CLASS_ORDER
        self.input_mean = OrderedDict({
            'RGB':  [109, 108, 102],
            'Acce': [1.598, 7.939, 1.104],
            'Gyro': [-0.039, -0.090, -0.004]
        })
        self.input_std = OrderedDict({
            'RGB':  [1, 1, 1],
            'Acce': [1.431, 2.831, 4.506],
            'Gyro': [3.299, 6.858, 2.558]
        })


class iDataEgo_TSN(iData_TSN):
    def __init__(self, model, modality, arch, train_list, test_list):
        super().__init__(model, modality, arch, train_list, test_list)
        self.class_order = DATAEGO_CLASS_ORDER
        self.input_mean = OrderedDict({
            'RGB':  [0, 0, 0],
            'Acce': [0, 0, 0],
            'Gyro': [0, 0, 0]
        })
        self.input_std = OrderedDict({
            'RGB':  [1, 1, 1],
            'Acce': [1, 1, 1],
            'Gyro': [1, 1, 1]
        })