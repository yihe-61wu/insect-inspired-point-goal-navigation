from math import pi
import numpy as np
from numpy import bool_, int64, ndarray


def normalize_angle(angle: ndarray) -> ndarray:
    angle %= 2 * pi
    if angle > pi: angle -= 2 * pi
    return angle


class queue:
    def __init__(self, queue_length, init_val=None):
        self.length = queue_length
        self.init_val = init_val
        self.reset()

    def __getitem__(self, index):
        return self.memory[index]

    def reset(self):
        self.memory = []
        if self.init_val is not None:
            self.update(self.init_val)

    def update(self, new_item):
        for _ in range(self.length + 1 - len(self.memory)):
            self.memory.append(new_item)
        self.memory.pop(0)
        return self.memory
