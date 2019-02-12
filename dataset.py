import os
import os.path
import torch.utils.data as data
from PIL import Image


# PREMISE: Images and masks are in the same directory as well as share the same filename
def create_dataset(root):
    imageList = [os.path.split(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
    return [(os.path.join(root, imageName + '.jpg'), os.path.join(root, imageName + '.png')) for imageName in imageList]

class DataMaskSet(data.Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.root = root
        self.image_list = create_dataset(root)
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image_path, mask_path = self.image_list[index]
        image = Image.open(image_path).convert('RGB')
        target = Image.open(mask_path).convert('L')
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_list)