import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm
import os

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')

bbox_dataset = []
with open('experiment/bbox_dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for ind_row,row in enumerate(reader):
        if ind_row==0:
            continue
        bbox_dataset.append([row[0],int(row[1]),int(row[2]),int(row[3]),int(row[4])])

img_dir = '../bird_dataset'

index_image = 0
for classe in tqdm(os.listdir(img_dir+'/train_images')):
    for f in tqdm(os.listdir(img_dir+'/train_images/'+classe)):
        if 'jpg' in f:
            image_orig = pil_loader(img_dir+'/train_images/'+classe + '/' + f)
        assert bbox_dataset[index_image][0] in f

        image_orig = np.array(image_orig, dtype=np.uint8)

        # Create figure and axes
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(image_orig)

        # Create a Rectangle patch
        bbox = bbox_dataset[index_image][1:]
        x0,y0,x1,y1 = bbox
        w = x1-x0
        h = y1-y0
        rect = patches.Rectangle((x0,y0),w,h,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()
        index_image += 1