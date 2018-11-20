import zipfile
import os
import PIL.Image as Image
from tqdm import tqdm
import csv
import numpy as np

import torchvision.transforms as transforms
import torch

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

# size: square 224 for VGG, 299 for inception v3
SIZE = 224

data_transforms = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

augmentation_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')

class ImageBoxDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform

        self.images = []
        self.classes = []
        self.classe_names = []
        self.bbox_dataset = []

        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for ind_row,row in enumerate(reader):
                if ind_row==0: # column names
                    continue
                x0,y0,x1,y1 = int(row[1]),int(row[2]),int(row[3]),int(row[4])
                xc,yc = (x1+x0)/2, (y1+y0)/2
                w,h = (x1-x0), (y1-y0)
                self.bbox_dataset.append([row[0],x0,y0,x1,y1,xc,yc,w,h]) # file box*4

        ind = 0
        for ind_classe, classe in enumerate(tqdm(os.listdir(root_dir))):
            self.classe_names.append(classe)
            for f in tqdm(os.listdir(root_dir+'/'+classe)):
                assert 'jpg' in f
                assert self.bbox_dataset[ind][0] in f
                # image = pil_loader(root_dir + '/' + classe + '/' + f)
                image = root_dir + '/' + classe + '/' + f
                self.images.append(image)
                self.classes.append(ind_classe)
                ind += 1

        assert len(self.images) == len(self.bbox_dataset)
        assert len(self.images) > 45*20
        assert len(self.classe_names) == 20, len(self.classe_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = pil_loader(self.images[idx])
        bbox = np.array(self.bbox_dataset[idx][1:9],dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        target = self.classes[idx]
        sample = [image,target,bbox]

        return sample

# def augmentation_transforms(img,points,flip):
#     [x0,y0,x1,y1] = points
#     w,h = img.size
#     cx,cy = int(w/2), int(h/2)

#     if flip:
#         img = transforms.functional.hflip(img)
#         x0,x1 = w-x1, w-x0

#     img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img)

#     img = data_transforms = transforms.Compose([
#         transforms.Resize((299, 299)),
#         transforms.ToTensor()
#     ])(img)

#     noise = Variable(img.data.new(img.size()).normal_(mean=0, stddev=1))
#     img = img+noise

#     img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])(img)

#     return img

