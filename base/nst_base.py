import torch
import random
from torch.utils.data import Dataset
from torch.nn.functional import pad
import numpy as np


class NST_dataset(Dataset):
    def __init__(self, path, transform=None, with_name=False, resolution=1):
        self.path = path
        self.files = list(path.glob('**/*.csv'))
        self.transform = transform
        self.with_name = with_name
        self.resolution = resolution

    def __len__(self):
        return len(list(self.path.glob('**/*.csv')))

    def __getitem__(self, idx):
        # x = np.load(str(self.files[idx]))
        # x = torch.from_numpy(str(self.files[idx])).float()
        x = np.loadtxt(self.files[idx], delimiter=',')
        # print(x)
        _fhr = x[:, 1]
        _uc = x[:, 2]

        _fhr = _fhr[::int(1/self.resolution)]
        _uc = _uc[::int(1/self.resolution)]
        x = np.stack([_fhr, _uc], axis=0)
        # print(x)
        label = int(self.files[idx].parts[-2])
        sample = {'input': x, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        if self.with_name:
            names = self.files[idx].stem.split('_')
            name = '{}_{}'.format(names[0], names[1])
            sample['name'] = name
        # print(sample)
        return sample


class OneHotEncoding(object):
    def __init__(self, num_classes):
        self.encoder = torch.eye(num_classes, dtype=torch.float)

    def __call__(self, sample):
        tensor, label = sample['input'], sample['label']
        print(label)
        label = self.encoder[label]

        sample = {'input': tensor, 'label': label}
        return sample


class GoTensor(object):
    def __call__(self, sample):
        inp = sample['input']
        inp = torch.from_numpy(inp).float()
        #         print(inp.size())

        label = sample['label']
        label = torch.as_tensor(label).float()
        # label = torch.as_tensor(label).long()

        sample = {'input': inp, 'label': label}
        # print(sample)
        return sample


class FrequencyMask(object):
    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, sample):
        tensor, label = sample['input'], sample['label']

        start = random.randrange(0, tensor.shape[1] - self.max_width)
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = 0

        sample = {'input': tensor, 'label': label}
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class TimeMask(object):
    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, sample):
        tensor, label = sample['input'], sample['label']

        start = random.randrange(0, tensor.shape[1] - self.max_width)
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, start:end] = tensor.mean()
        else:
            tensor[:, start:end] = 0

        sample = {'input': tensor, 'label': label}
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class RandomShift(object):
    def __init__(self, max_dist):
        self.max_dist = max_dist

    def __call__(self, sample):
        tensor, label = sample['input'], sample['label']

        dist = random.randrange(-self.max_dist, self.max_dist)

        if dist >= 0:
            tensor = pad(tensor, (0, dist), "constant", 0)

        else:
            tensor = tensor[:, :dist].contiguous()
            pass

        sample = {'input':tensor, 'label':label}
        return sample


class Jittering(object):
    def __init__(self, center, sigma):
        self.sigma = sigma
        self.center = center

    def __call__(self, sample):
        tensor, label = sample['input'], sample['label']
        rand = np.random.normal(self.center, self.sigma, (tensor.shape[0], tensor.shape[1]))
        inp = torch.from_numpy(rand).float()

        tensor += tensor + inp

        sample = {'input': tensor, 'label': label}
        return sample


class zero_pad(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        tensor, label = sample['input'], sample['label']
        tensor = pad(tensor, (self.size-tensor.shape[1], 0), "constant", 0)
        sample = {'input': tensor, 'label': label}
        return sample


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss