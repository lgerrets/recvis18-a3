import zipfile
import os
import PIL.Image as Image

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

augmentation_transforms = transforms.Compose([
	transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

# def augmentation_transforms(img,points,flip):
#     [x0,y0,x1,y1] = points
#     w,h = img.size
#     cx,cy = int(w/2), int(h/2)

#     if flip:
#         img = transforms.functional.hflip(img)
#         x0,x1 = w-x1, w-x0

#     img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(img)

#     img = data_transforms = transforms.Compose([
# 	    transforms.Resize((299, 299)),
# 	    transforms.ToTensor()
# 	])(img)

#     noise = Variable(img.data.new(img.size()).normal_(mean=0, stddev=1))
#     img = img+noise

# 	img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])(img)

#     return img

