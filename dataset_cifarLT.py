import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from utils import CIFAR10Pair
np.random.seed(1234)
torch.manual_seed(1234)

class CIFAR10LT(CIFAR10Pair):
    cls_num = 10

    def __init__(self, root, imb_factor, imb_type='exp', rand_number=0, train=True,
                 mem=False, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10LT, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(1234)

        # taken from std. implemntation
        # self.mode = 'train' if train == True else 'test'
        # self.mem = mem
        # MEAN = [0.4914, 0.4822, 0.4465]
        # STD  = [0.2023, 0.1994, 0.2010]
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        # self.augment_transform = transforms.Compose([transforms.RandomResizedCrop(32),
        #                                             transforms.RandomHorizontalFlip(p=0.5),
        #                                             transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #                                             transforms.RandomGrayscale(p=0.2)
        #                                             ])
        

        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

'''
    def __getitem__(self, index):
        sample = {}
        image = Image.fromarray(self.data[index]).convert('RGB')
        image_ = self.transform(image)
        label = self.targets[index]

        # if self.mem == True:
        #     image = self.transform(image)
        #     return image, image, label

        if self.mode == 'train':
            image_ = self.transform(image)
            image_x1 = self.transform(self.augment_transform(image))
            image_x2 = self.transform(self.augment_transform(image))

            return image_x1, image_x2, image_, label
            # sample = {'image' : image_, 'image_x1' : image_x1, 'image_x2' : image_x2, 'label' : label}

        elif self.mode == 'test':
            image = self.transform(image)
            return image, label
            # sample = {'image' : image, 'label' : label}

        return sample
'''    

class CIFAR100LT(CIFAR10LT):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    '''for CIFAR-100'''

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


def get_dataloader_cifarLT(args):
    if args.dataset == 'CIFAR10':
        trainset = CIFAR10LT(root='./../data', train=True, download=True, imb_factor=args.imbl_factor)
        testset = CIFAR10LT(root='./../data', train=False, download=True, imb_factor=1)
    elif args.dataset == 'CIFAR100':
        trainset = CIFAR100LT(root='./../data', train=True, download=True, imb_factor=args.imbl_factor)
        testset = CIFAR100LT(root='./../data', train=False, download=True, imb_factor=1)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    return trainloader, testloader


if __name__ == '__main__':
    trainset = CIFAR10LT(root='./../data', train=True, imb_factor=0.01)    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    
