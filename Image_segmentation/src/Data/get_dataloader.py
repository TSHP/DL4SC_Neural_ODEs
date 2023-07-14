import os
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

class CustomVOCDataset(Dataset):
    def __init__(self, voc_dataset, transform_image=None, transform_mask=None):
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.voc_dataset = voc_dataset

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, mask = self.voc_dataset[idx]

        if self.transform_image is not None:
            image = self.transform_image(image)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        return image, mask
        
def encode(x):
    x[x==255] = 0
    x = F.one_hot(x.to(torch.int64), 21).permute(0,3,1,2).to(torch.float)
    x = torch.squeeze(x)
    return x

def get_dataloader(voc_dataset, out_size=32, batch_size=32):

    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((out_size, out_size), interpolation= transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        lambda x: encode(x) 
    ])

    dataset = CustomVOCDataset(voc_dataset, transform_image=transform_image,
                            transform_mask=transform_mask)

    VOC_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return VOC_data_loader
