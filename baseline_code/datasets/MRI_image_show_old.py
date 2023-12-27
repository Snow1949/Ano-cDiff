from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import torch as th

path1 = "/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_train/patient_BraTS2021_01259" \
        "/slice_54.npz"
path2 = "/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_val/patient_BraTS2021_01112" \
        "/slice_54.npz"
path3 = "/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_test/patient_BraTS2021_01609" \
        "/slice_54.npz"
data1 = np.load(path1)
print(type(data1))
print(data1.files)

# plt.imshow(data1)
img_data1 = data1['x']
print("train_shape:{}".format(img_data1.shape))         # train_shape:(1, 4, 128, 128)
# print(img_data1)
seg_data1 = data1['y']
print(type(seg_data1))

result_img0 = img_data1[0, 0, ...]
print('type(input_img0):{}'.format(type(result_img0)))
print('shape(input_img0):{}'.format(result_img0.shape))          # torch.Size([64, 64])
# plt.imshow(result_img0, cmap='inferno')
plt.imshow(result_img0, cmap='gray')
# plt.savefig('./figure/_ResUp_Blank_again_nolr_' + name + '.png')
path = '/chenxue/Diff-SCM-main/diff_scm/pictures/'
plt.savefig(path + 'train_image_0724.jpg')
# images.save(path + 'train_image.jpg')
plt.show()
result_img1 = img_data1[:, 1, ...]
result_img2 = img_data1[:, 2, ...]
result_img3 = img_data1[:, 3, ...]

# images_tensor = th.cat([result_img0, result_img1, result_img2, result_img3], dim=1)
# print('images.size:{}'.format(images_tensor.size()))       # torch.size([64, 320])
# images_numpy = images_tensor.cpu().numpy()
# images = Image.fromarray(np.uint8(images_numpy))
# save_image(images, 'input_png')
# images.save('result_img/' + str(key) + '/' + str(ith) + '.jpg')


print("seg_shape:{}".format(seg_data1.shape))         # ] seg_shape:(1, 1, 128, 128)
# print(seg_data1)
# images = Image.fromarray(np.uint8(img_data1))
# save_image(images, 'input_png')
# path = '/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_train/patient_BraTS2021_01259/'
# images.save(path + 'train_image.jpg')


data2 = np.load(path2)
# print("val_shape:{}".format(data2.size))
# plt.imshow(data2)
# img_data2 = data2['img']
# seg_data2 = data2['seg']
# show the details
# data1.files
data3 = np.load(path3)
# print("test.shape:{}".format(data3.size))
# plt.imshow(data3)

# MRI_image_show.py
# cd /chenxue/Diff-SCM-main/diff_scm/datasets && python -u MRI_image_show.py
