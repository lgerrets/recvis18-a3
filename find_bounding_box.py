import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
from torchvision import datasets
import numpy as np


import torch


parser = argparse.ArgumentParser(description='Generate the bounding box dataset')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/bbox_dataset.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model)
from model import createModel
model = createModel()
model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

from data import data_transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as image:
            return image.convert('RGB')

windows_ratios = []
for pow_scale in range(3):
    scale = (2/3)**pow_scale
    stride = scale/3
    topleft_x = 0
    while topleft_x <= 1-scale:
        topleft_y = 0
        while topleft_y <= 1-scale:
            bottomright_x = topleft_x+scale
            bottomright_y = topleft_y+scale
            windows_ratios.append([topleft_x,topleft_y,bottomright_x,bottomright_y])
            topleft_y += stride
        topleft_x += stride
# print(len(windows_ratios))


output_file = open(args.outfile, "w")
output_file.write("ImageId,BBoxx0,BBoxy0,BBoxx1,BBoxy1\n")

model.eval()
classes = os.listdir(args.data + '/train_images')
classes_real = [5,19,16,6,17,18,7,13,2,11,12,9,8,0,1,4,3,10,15,14]
for ind_classe,classe in enumerate(classes):
    ind_classe_real = classes_real[ind_classe]
    predictions = []
    for f in tqdm(os.listdir(args.data+'/train_images/'+classe)):
        if 'jpg' in f:
            image_orig = pil_loader(args.data+'/train_images/'+classe + '/' + f)

        w,h = image_orig.size
        predscores = []
        for ind_window, window_ratio in enumerate(windows_ratios):
            # get window and transform
            x0,y0,x1,y1 = int(window_ratio[0]*w), int(window_ratio[1]*h), int(window_ratio[2]*w), int(window_ratio[3]*h)
            data_windowed = image_orig.transform(size=(x1-x0,y1-y0),method=Image.EXTENT,data=(x0,y0,x1,y1),resample=Image.BILINEAR)
            data_windowed = data_transforms(data_windowed)
            data_windowed = data_windowed.view(1, data_windowed.size(0), data_windowed.size(1), data_windowed.size(2))

            if use_cuda:
                data_windowed = data_windowed.cuda()
            output = model(data_windowed).data[0]

            scores = output.data
            pred = scores.argmax()
            scores = np.sort(scores)
            predscore = scores[-1]
            diffscore = scores[-1]-scores[-2]
            classe_score = output.data[ind_classe_real]

            print(ind_window,scores[-1],scores[-2],pred,ind_classe_real)

            if ind_classe_real != pred:
                predscores.append(-np.inf)
            else:
                predscores.append(predscore)
            
        ind_best_window = np.argmax(predscores)
        window_ratio = windows_ratios[ind_best_window]
        x0,y0,x1,y1 = int(window_ratio[0]*w), int(window_ratio[1]*h), int(window_ratio[2]*w), int(window_ratio[3]*h)
        output_file.write("%s,%d,%d,%d,%d,%.2f\n" % (f[:-4],x0,y0,x1,y1))

print("Succesfully wrote " + args.outfile)


