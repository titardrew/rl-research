import os
from os import path
from PIL import Image
import numpy as np
import json
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import torch

torch.set_default_dtype(torch.float64)

class PlanarDatasetN(Dataset):
    width = 40
    height = 40
    action_dim = 2

    def __init__(self, dir):
        self.dir = dir
        self.n_steps = None
        with open(path.join(dir, 'data.json')) as f:
            self._data = json.load(f)
        self._process()

    def __len__(self):
        return len(self._data['samples'])

    def __getitem__(self, index):
        return self._processed[index]

    @staticmethod
    def _process_image(img):
        return ToTensor()((img.convert('L').
                           resize((PlanarDataset.width,
                                   PlanarDataset.height))))

    def _process(self):
        preprocessed_file = os.path.join(self.dir, 'processed.pkl')
        if True or not os.path.exists(preprocessed_file):
            processed = []
            self.n_steps = None
            for sample in tqdm(self._data['samples'], desc='processing data'):
                obs0 = Image.open(os.path.join(self.dir, sample['observation_files'][0]))

                processed_sample = [self._process_image(obs0)]
                n_steps = 0
                for obs_file, control in zip(sample['observation_files'][1:], sample['controls']):
                    obs = Image.open(os.path.join(self.dir, obs_file))
                    processed_sample += [np.asarray(control), self._process_image(obs)]
                    n_steps += 1

                if self.n_steps is None:
                    self.n_steps = n_steps
                assert self.n_steps == n_steps
                processed.append(tuple(processed_sample))

            with open(preprocessed_file, 'wb') as f:
                pickle.dump(processed, f)
            self._processed = processed
        else:
            with open(preprocessed_file, 'rb') as f:
                self._processed = pickle.load(f)
