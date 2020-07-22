from collections import deque
import random

import torch


class RolloutHistory:
    def __init__(self, maxsize):
        self.history = deque(maxlen=maxsize)

    @property
    def maxlen(self):
        return self.history.maxlen

    @property
    def full(self):
        return self.current_size >= self.maxsize

    def __len__(self):
        return self.history.__len__()

    def __getitem__(self, item):
        return self.history.__getitem__(item)

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Samples a batch randomly in prior experiences

        :arg batch_size: size of the batch to sample"""

        obs_shape = self.history[0].observation_space.shape

        history_indexes = []
        for i in range(batch_size):
            history_indexes.append(random.randint(0, len(self) - 1))

        batch = []

        for index in history_indexes :
            sample = random.choice(self.history[index].observations)
            batch.append(sample)

        return torch.Tensor(batch)


    def append(self,item):
        self.history.append(item)
