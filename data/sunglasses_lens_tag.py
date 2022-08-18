import os
import csv
import shutil

dataset_path = os.path.join(os.getcwd(), 'CelebAMask-HQ')
glasses_dataset_path = os.path.join(dataset_path, 'glasses_only')
csv_file = os.path.join(dataset_path, 'glasses_type.csv')
seg_dataset_path = os.path.join(dataset_path, 'CelebAMask-HQ-mask-anno-expanded/')

im_dir_t = 'image_train'
im_dir_v = 'image_val'
seg_dir_t = 'label_train'
seg_dir_v = 'label_val'

img_trainset_path = os.path.join(dataset_path, im_dir_t)
seg_trainset_path = os.path.join(dataset_path, seg_dir_t)
img_valset_path = os.path.join(dataset_path, im_dir_v)
seg_valset_path = os.path.join(dataset_path, seg_dir_v)

g_img_trainset_path = os.path.join(glasses_dataset_path, im_dir_t)
g_seg_trainset_path = os.path.join(glasses_dataset_path, seg_dir_t)
g_img_valset_path = os.path.join(glasses_dataset_path, im_dir_v)
g_seg_valset_path = os.path.join(glasses_dataset_path, seg_dir_v)

print(csv_file, seg_dataset_path)

dir_list = list()
dict_list = dict()

for folder in os.listdir(seg_dataset_path):
    if len(folder) < 3:
        search_dir = os.path.join(seg_dataset_path, folder)
        dict_list[folder] = os.listdir(search_dir)


assert os.path.isdir(img_trainset_path)
assert os.path.isdir(seg_trainset_path)
assert os.path.isdir(img_valset_path)
assert os.path.isdir(seg_valset_path)

#os.mkdir(g_img_trainset_path)
os.mkdir(g_seg_trainset_path)
#os.mkdir(g_img_valset_path)
os.mkdir(g_seg_valset_path)

file = open(csv_file)
csvreader = csv.reader(file)

i,j = 0,0
count = 0
for row in csvreader:
    n = int(row[0].split(".")[0])

    if n < 28000:
        img_input_path = img_trainset_path
        img_output_path = g_img_trainset_path
        seg_input_path = seg_trainset_path
        seg_output_path = g_seg_trainset_path
    else:
        img_input_path = img_valset_path
        img_output_path = g_img_valset_path
        seg_input_path = seg_valset_path
        seg_output_path = g_seg_valset_path

    im_input_path = os.path.join(img_input_path, f'{n}.jpg')
    im_output_path = os.path.join(img_output_path, f'{n}.jpg')
    sg_input_path = os.path.join(seg_input_path, f'{n}.png')
    sg_output_path = os.path.join(seg_output_path, f'{n}.png')
    #shutil.copyfile(im_input_path, im_output_path)
    shutil.copyfile(sg_input_path, sg_output_path)
    # name = '{:0>5}_eye_lens.png'.format(row[0].split(".")[0])
    # if row[1] == 'Sunglasses':
    #     for folder,names in dict_list.items():
    #         if name in names:
    #             old_name = os.path.join(seg_dataset_path, folder, name)
    #             new_name = os.path.join(seg_dataset_path, folder, name.split("_")[0]+'_eye_sunlens.png')
    #             if os.path.exists(old_name):
    #                 os.rename(old_name, new_name)
    #                 i+=1
    #             else:
    #                 print(name, row, old_name)
    #             break
    # elif row[1] == 'Eyeglasses':
    #     j+=1
    #     continue
    # else:
    #     print(row)

file.close()
print(i,j,i+j)
print(count)