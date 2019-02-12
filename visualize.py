import numpy as np
import matplotlib.pyplot as plt
import torchvision

class Board:

    def __init__(self):
        super().__init__()

    def imshow(self, inp, title=None):
        inp = inp.numpy.transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)

    def show_image(self, image, title=None):
        out = torchvision.utils.make_grid(image)
        self.imshow(image, title)

