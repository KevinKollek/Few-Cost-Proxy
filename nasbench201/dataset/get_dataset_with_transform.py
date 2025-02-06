import gzip
import pickle
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from nasbench201.configs.config_utils import load_config

# Number of classes for each dataset
Dataset2Class = {
    'cifar10-valid': 10,
    'syn_cifar10': 10,
    'cifar100': 100,
    'scifar100': 100
}

# Mean and std for each dataset
DATASET_STATS = {
    'cifar10-valid': {
        'mean': [125.3 / 255, 123.0 / 255, 113.9 / 255],
        'std': [63.0 / 255, 62.1 / 255, 66.7 / 255]
    },
    'cifar100': {
        'mean': [129.3 / 255, 124.1 / 255, 112.4 / 255],
        'std': [68.2 / 255, 65.4 / 255, 70.4 / 255]
    },
    'syn_cifar10': {
        'mean': [0.0, 0.0, 0.0],
        'std': [1.0, 1.0, 1.0]
    },
    'scifar100': {
        'mean': [51.6 / 255, 56.2 / 255, 57.9 / 255],
        'std': [79.7 / 255, 77.4 / 255, 74.25 / 255]
    }
}


# Synthetic CIFAR-10 Dataset
class SyntheticCIFAR10(data_utils.Dataset):
    def __init__(self, data_dict):
        """
        Expects data_dict with:
          - 'image': FloatTensor of shape [N, H, W, C]
          - 'label': LongTensor of shape [N]
        """
        self.images = data_dict['image']  # [N, H, W, C]
        self.labels = data_dict['label'].long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert from [H, W, C] to [C, H, W]
        image = self.images[idx].permute(2, 0, 1)
        label = self.labels[idx]
        return image, label


# Spherical CIFAR-100 (scifar100) Loader
def load_scifar100_data(gz_path):
    """
    Loads 'scifar100' from a gzip file containing a dict:
      dataset = {
        "train": {
           "images": numpy array [N_train, 3, H, W],
           "labels": numpy array [N_train]
        },
        "test": {
           "images": numpy array [N_test, 3, H, W],
           "labels": numpy array [N_test]
        }
      }
    """
    with gzip.open(gz_path, 'rb') as f:
        dataset = pickle.load(f)

    # Convert to tensors and normalize from [0..255] to [0..1]
    train_images = torch.from_numpy(dataset["train"]["images"].astype(np.float32)) / 255.0
    train_labels = torch.from_numpy(dataset["train"]["labels"].astype(np.int64))
    test_images = torch.from_numpy(dataset["test"]["images"].astype(np.float32)) / 255.0
    test_labels = torch.from_numpy(dataset["test"]["labels"].astype(np.int64))

    train_data = data_utils.TensorDataset(train_images, train_labels)
    test_data = data_utils.TensorDataset(test_images, test_labels)

    xshape = (1, 3, 60, 60)  # example shape for scifar100 images
    class_num = Dataset2Class['scifar100']
    return train_data, test_data, xshape, class_num


def get_datasets(dataset_name, root_or_path):
    """
    Returns (train_data, test_data, xshape, class_num)
      - train_data, test_data: PyTorch Dataset objects
      - xshape: shape of single input sample: (1, C, H, W)
      - class_num: number of classes
    """
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    mean = DATASET_STATS[dataset_name]['mean']
    std = DATASET_STATS[dataset_name]['std']

    # Decide transforms
    if dataset_name in ['cifar10-valid', 'cifar100']:
        # Real CIFAR with data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        xshape = (1, 3, 32, 32)

        if dataset_name == 'cifar10-valid':
            train_data = dset.CIFAR10(root_or_path, train=True, transform=train_transform, download=True)
            test_data = dset.CIFAR10(root_or_path, train=False, transform=test_transform, download=True)
            assert len(train_data) == 50000 and len(test_data) == 10000
        else:
            # cifar100
            train_data = dset.CIFAR100(root_or_path, train=True, transform=train_transform, download=True)
            test_data = dset.CIFAR100(root_or_path, train=False, transform=test_transform, download=True)
            assert len(train_data) == 50000 and len(test_data) == 10000

    elif dataset_name == 'syn_cifar10':
        # Synthetic CIFAR-10
        xshape = (1, 3, 32, 32)

        data = torch.load(root_or_path)  # e.g. 'dataset/synthetic_cifar10_1.pt'
        train_dict = data['train']
        test_dict = data['test']
        train_data = SyntheticCIFAR10(train_dict)
        test_data = SyntheticCIFAR10(test_dict)
        assert len(train_data) == 50000 and len(test_data) == 10000

    elif dataset_name == 'scifar100':
        # Spherical CIFAR-100
        train_data, test_data, xshape, _ = load_scifar100_data(root_or_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    class_num = Dataset2Class[dataset_name]
    return train_data, test_data, xshape, class_num


def select_dataset(dataset_name, workers):
    """
    Returns:
      train_loader, ValLoaders (dict), config
    Where ValLoaders can have keys like:
      'ori-test' -> the original test set loader
      'x-valid'  -> validation subset (when we do a real train/valid split)
      'x-test'   -> special test subsets for CIFAR-100, etc.
    """

    if dataset_name in ['cifar10-valid', 'cifar100']:
        xpath = 'nasbench201/dataset/cifar.python'
    elif dataset_name == 'syn_cifar10':
        xpath = 'nasbench201/dataset/synthetic/synthetic_cifar10_1.pt'
    elif dataset_name == 'scifar100':
        xpath = 'nasbench201/dataset/spherical/s2_cifar100.gz'
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    # For scifar100, we use a different loader
    if dataset_name == 'scifar100':
        train_data, test_data, xshape, class_num = load_scifar100_data(xpath)
    else:
        train_data, test_data, xshape, class_num = get_datasets(dataset_name, xpath)

    # Decide config path & split file
    if dataset_name in ['cifar10-valid', 'cifar100']:
        config_path = 'nasbench201/configs/CIFAR_eval.config'
        split_info = load_config('nasbench201/configs/cifar-split.txt', None)
    elif dataset_name == 'syn_cifar10':
        config_path = 'nasbench201/configs/Synthetic_CIFAR.config'
        split_info = load_config('nasbench201/configs/cifar-split.txt', None)
    elif dataset_name == 'scifar100':
        config_path = 'nasbench201/configs/Spherical_CIFAR100.config'
        split_info = load_config('nasbench201/configs/cifar-split.txt', None)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    # Load the config with class_num + xshape
    config = load_config(config_path, {
        'class_num': class_num,
        'xshape': xshape
    })

    if dataset_name in ['cifar10-valid', 'cifar100']:
        # The official test set loader:
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True
        )

        # We do a subset sampler for train/validation
        assert len(train_data) == len(split_info.train) + len(split_info.valid), \
            f"Split mismatch: {len(train_data)} vs {len(split_info.train)} + {len(split_info.valid)}"

        # For validation, we usually disable data augmentation => copy dataset and replace transform
        train_data_v2 = copy.deepcopy(train_data)
        # Use the same transform as test_data
        # (only if train_data/test_data have a `.transform` attribute)
        if hasattr(test_data, 'transform') and hasattr(train_data_v2, 'transform'):
            train_data_v2.transform = test_data.transform

        # SubsetRandomSamplers for train & valid
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.train),
            num_workers=workers,
            pin_memory=True
        )
        valid_loader = torch.utils.data.DataLoader(
            train_data_v2,
            batch_size=config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.valid),
            num_workers=workers,
            pin_memory=True
        )

        # Collect validation/test loaders in a dict
        ValLoaders = {
            'ori-test': test_loader,
            'x-valid': valid_loader
        }

        if dataset_name == 'cifar100':
            # Example of extra splits
            cifar100_splits = load_config('nasbench201/configs/cifar100-test-split.txt', None)
            # Add x-valid / x-test from the official test set
            ValLoaders['x-valid'] = torch.utils.data.DataLoader(
                test_data,
                batch_size=config.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_splits.xvalid),
                num_workers=workers,
                pin_memory=True
            )
            ValLoaders['x-test'] = torch.utils.data.DataLoader(
                test_data,
                batch_size=config.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_splits.xtest),
                num_workers=workers,
                pin_memory=True
            )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=config.batch_size, shuffle=True,
            num_workers=workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False,
            num_workers=workers, pin_memory=True
        )
        # Only one validation loader: the original test set
        ValLoaders = {'ori-test': test_loader}

    return train_loader, ValLoaders, config
