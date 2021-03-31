import os
import PIL.Image as Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


img_path = "/home/admin/segmentation/task3/data/vaihingen/train/cropped/images/train"
label_path = "/home/admin/segmentation/task3/data/vaihingen/train/cropped/labels/train"
boundary_path = "/home/admin/segmentation/task3/boundary/train"
grad_path = "/home/admin/segmentation/task3/grad/train"


img_files = sorted(os.listdir(img_path))
label_files = sorted(os.listdir(label_path))
boundary_files = sorted(os.listdir(boundary_path))
grad_files = sorted(os.listdir(grad_path))


idx = 10
image = img_files[idx]
label = label_files[idx]
boundary = boundary_files[idx]
grad = grad_files[idx]
# print(image)
image = Image.open(f"{img_path}/{image}")
# print(image)
label = Image.open(f"{label_path}/{label}")
grad = Image.open(f"{grad_path}/{grad}")
boundary = Image.open(f"{boundary_path}/{boundary}")
# print((np.array(grad)==0).sum())

plt.imshow(image)
plt.savefig("output_img.png")

plt.imshow(label)
plt.savefig("output_label.png")
plt.imshow(grad)
plt.savefig("output_grad.png")
plt.imshow(boundary)
plt.savefig("output_boundary.png")
