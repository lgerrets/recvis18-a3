import numpy as np
from PIL import ImageFilter
import PIL.Image as img
from tqdm import tqdm
import os


input_dir = '../bird_dataset/train_images'
output_dir = '../bird_dataset/train_aug'

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with img.open(f) as image:
			return image.convert('RGB')

for classe in tqdm(os.listdir(input_dir)):
	os.mkdir(output_dir+'/'+classe)
	for f in tqdm(os.listdir(input_dir+'/'+classe)):
		if 'jpg' in f:
			image_orig = pil_loader(input_dir + '/' + classe + '/' + f)

			# add flipped
			flipped_image = image_orig.transpose(img.FLIP_LEFT_RIGHT)
			images = [image_orig,flipped_image]

			# add rotations
			to_append = []
			for image in images:
				for it in range(3):
					angle = np.random.randint(-40,40)
					resample = img.NEAREST # NEAREST / BILINEAR / BICUBIC
					image_rot = image.rotate(angle,resample)
					to_append += [image_rot]
			images += to_append

			# noise all
			for image in images:
				image.filter(ImageFilter.GaussianBlur(5))

			# save all
			for ind,image in enumerate(images):
				image.save(output_dir+'/'+classe+'/'+str(ind)+f)

		


