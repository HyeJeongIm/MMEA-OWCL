import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from .mydataset import MyDataSet
from utils.transforms import ArrayToTensor, DataStack, GroupNormalize, IdentityTransform, ImgStack, ToTorchFormatTensor, GroupScale, GroupCenterCrop
import torch.utils.data as data


COMMON_CLASS_ORDER = [
    26, 14, 23, 4, 11, 25, 31, 10,
    29, 5, 6, 9, 17, 22, 2, 19,
    13, 1, 21, 16, 8, 3, 27, 28,
    15, 30, 0, 7, 12, 18, 20, 24
]


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iUESTC_MMEA_TBN(iData):
    use_path = False
    class_order = COMMON_CLASS_ORDER

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


class iUESTC_MMEA_TSN(iData):
    use_path = False
    class_order = COMMON_CLASS_ORDER

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