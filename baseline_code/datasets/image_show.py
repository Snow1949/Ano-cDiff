from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import torch as th
# cd /chenxue/paper3/Ano-cDiff/optimal_model/datasets && python -u data_preprocessing.py
tumor1 = "/chenxue/paper3/Ano-cDiff/optimal_model/datasets/data/brats2021_64x64/npy_train/patient_BraTS2021_01125" \
        "/slice_45.npz"
tumor2 = "/chenxue/paper3/Ano-cDiff/optimal_model/datasets/data/brats2021_64x64/npy_train/patient_BraTS2021_00838" \
        "/slice_10.npz"
tumor3 = "/chenxue/paper3/Ano-cDiff/optimal_model/datasets/data/brats2021_64x64/npy_train/patient_BraTS2021_01205" \
        "/slice_27.npz"
healthy1 = "/chenxue/paper3/Ano-cDiff/optimal_model/datasets/data/brats2021_64x64/npy_train/patient_BraTS2021_00103" \
        "/slice_107.npz"
healthy2 = "/chenxue/paper3/Ano-cDiff/optimal_model/datasets/data/brats2021_64x64/npy_train/patient_BraTS2021_01294" \
        "/slice_98.npz"
healthy3 = "/chenxue/paper3/Ano-cDiff/optimal_model/datasets/data/brats2021_64x64/npy_train/patient_BraTS2021_00210" \
        "/slice_111.npz"

healthy1_np = np.load(healthy1)
healthy2_np = np.load(healthy2)
healthy3_np = np.load(healthy3)
tumor1_np = np.load(tumor1)
tumor2_np = np.load(tumor2)
tumor3_np = np.load(tumor3)
healthy1_img0 = healthy1_np['x'][0, 0, ...]
healthy1_img1 = healthy1_np['x'][0, 1, ...]
healthy1_img2 = healthy1_np['x'][0, 2, ...]
healthy1_img3 = healthy1_np['x'][0, 3, ...]
healthy1_img4 = healthy1_np['y'][0, 0, ...]
healthy2_img0 = healthy2_np['x'][0, 0, ...]
healthy2_img1 = healthy2_np['x'][0, 1, ...]
healthy2_img2 = healthy2_np['x'][0, 2, ...]
healthy2_img3 = healthy2_np['x'][0, 3, ...]
healthy2_img4 = healthy2_np['y'][0, 0, ...]
healthy3_img0 = healthy3_np['x'][0, 0, ...]
healthy3_img1 = healthy3_np['x'][0, 1, ...]
healthy3_img2 = healthy3_np['x'][0, 2, ...]
healthy3_img3 = healthy3_np['x'][0, 3, ...]
healthy3_img4 = healthy3_np['y'][0, 0, ...]
# print('type(healthy1_img0):{}'.format(type(healthy1_img0)))
# print('shape(healthy1_img0):{}'.format(healthy1_img0.shape))          # torch.Size([64, 64])
# print('type(healthy1_img4):{}'.format(type(healthy1_img4)))
# print('shape(healthy1_img4):{}'.format(healthy1_img4.shape))          # torch.Size([64, 64])
tumor1_img0 = tumor1_np['x'][0, 0, ...]
tumor1_img1 = tumor1_np['x'][0, 1, ...]
tumor1_img2 = tumor1_np['x'][0, 2, ...]
tumor1_img3 = tumor1_np['x'][0, 3, ...]
tumor1_img4 = tumor1_np['y'][0, 0, ...]
tumor2_img0 = tumor2_np['x'][0, 0, ...]
tumor2_img1 = tumor2_np['x'][0, 1, ...]
tumor2_img2 = tumor2_np['x'][0, 2, ...]
tumor2_img3 = tumor2_np['x'][0, 3, ...]
tumor2_img4 = tumor2_np['y'][0, 0, ...]
tumor3_img0 = tumor3_np['x'][0, 0, ...]
tumor3_img1 = tumor3_np['x'][0, 1, ...]
tumor3_img2 = tumor3_np['x'][0, 2, ...]
tumor3_img3 = tumor3_np['x'][0, 3, ...]
tumor3_img4 = tumor3_np['y'][0, 0, ...]


plt.subplot(6, 5, 1)
plt.imshow(tumor1_img0, cmap='gray')
plt.subplot(6, 5, 2)
plt.imshow(tumor1_img1, cmap='gray')
plt.subplot(6, 5, 3)
plt.imshow(tumor1_img2, cmap='gray')
plt.subplot(6, 5, 4)
plt.imshow(tumor1_img3, cmap='gray')
plt.subplot(6, 5, 5)
plt.imshow(tumor1_img4, cmap='gray')
plt.subplot(6, 5, 6)
plt.imshow(tumor2_img0, cmap='gray')
plt.subplot(6, 5, 7)
plt.imshow(tumor2_img1, cmap='gray')
plt.subplot(6, 5, 8)
plt.imshow(tumor2_img2, cmap='gray')
plt.subplot(6, 5, 9)
plt.imshow(tumor2_img3, cmap='gray')
plt.subplot(6, 5, 10)
plt.imshow(tumor2_img4, cmap='gray')
plt.subplot(6, 5, 11)
plt.imshow(tumor3_img0, cmap='gray')
plt.subplot(6, 5, 12)
plt.imshow(tumor3_img1, cmap='gray')
plt.subplot(6, 5, 13)
plt.imshow(tumor3_img2, cmap='gray')
plt.subplot(6, 5, 14)
plt.imshow(tumor3_img3, cmap='gray')
plt.subplot(6, 5, 15)
plt.imshow(tumor3_img4, cmap='gray')

plt.subplot(6, 5, 16)
plt.imshow(healthy1_img0, cmap='gray')
plt.subplot(6, 5, 17)
plt.imshow(healthy1_img1, cmap='gray')
plt.subplot(6, 5, 18)
plt.imshow(healthy1_img2, cmap='gray')
plt.subplot(6, 5, 19)
plt.imshow(healthy1_img3, cmap='gray')
plt.subplot(6, 5, 20)
plt.imshow(healthy1_img4, cmap='gray')
plt.subplot(6, 5, 21)
plt.imshow(healthy2_img0, cmap='gray')
plt.subplot(6, 5, 22)
plt.imshow(healthy2_img1, cmap='gray')
plt.subplot(6, 5, 23)
plt.imshow(healthy2_img2, cmap='gray')
plt.subplot(6, 5, 24)
plt.imshow(healthy2_img3, cmap='gray')
plt.subplot(6, 5, 25)
plt.imshow(healthy2_img4, cmap='gray')
plt.subplot(6, 5, 26)
plt.imshow(healthy3_img0, cmap='gray')
plt.subplot(6, 5, 27)
plt.imshow(healthy3_img1, cmap='gray')
plt.subplot(6, 5, 28)
plt.imshow(healthy3_img2, cmap='gray')
plt.subplot(6, 5, 29)
plt.imshow(healthy3_img3, cmap='gray')
plt.subplot(6, 5, 30)
plt.imshow(healthy3_img4, cmap='gray')

plt.title('example_image')
# plt.imshow(healthy1_img0, cmap='inferno')
path = '/chenxue/Diff-SCM-main/diff_scm/pictures/'
plt.savefig(path + 'example_image.jpg')
plt.show()




# # plt.imshow(data1)
# img_data1 = data1['x']
# print("train_shape:{}".format(img_data1.shape))         # train_shape:(1, 4, 128, 128)
# # print(img_data1)
# seg_data1 = data1['y']
# print(type(seg_data1))
#
#
# # result_img0 = img_data1[0, 0, ...]
# # print('type(input_img0):{}'.format(type(result_img0)))
# # print('shape(input_img0):{}'.format(result_img0.shape))          # torch.Size([64, 64])
# plt.imshow(result_img0, cmap='inferno')
# path = '/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_train/patient_BraTS2021_01259/'
# plt.savefig(path + 'train_image.jpg')
# plt.show()
#
# result_img1 = img_data1[:, 1, ...]
# result_img2 = img_data1[:, 2, ...]
# result_img3 = img_data1[:, 3, ...]
#
# # images_tensor = th.cat([result_img0, result_img1, result_img2, result_img3], dim=1)
# # print('images.size:{}'.format(images_tensor.size()))       # torch.size([64, 320])
# # images_numpy = images_tensor.cpu().numpy()
# # images = Image.fromarray(np.uint8(images_numpy))
# # save_image(images, 'input_png')
# # images.save('result_img/' + str(key) + '/' + str(ith) + '.jpg')
#
#
# print("seg_shape:{}".format(seg_data1.shape))         # ] seg_shape:(1, 1, 128, 128)
# # print(seg_data1)
# # images = Image.fromarray(np.uint8(img_data1))
# # save_image(images, 'input_png')
# # path = '/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_train/patient_BraTS2021_01259/'
# # images.save(path + 'train_image.jpg')
#
#
# data2 = np.load(path2)
# # print("val_shape:{}".format(data2.size))
# # plt.imshow(data2)
# # img_data2 = data2['img']
# # seg_data2 = data2['seg']
# # show the details
# # data1.files
# data3 = np.load(path3)
# # print("test.shape:{}".format(data3.size))
# # plt.imshow(data3)
#
# # MRI_image_show.py
# # cd /chenxue/Diff-SCM-main/diff_scm/datasets && python -u MRI_image_show.py
