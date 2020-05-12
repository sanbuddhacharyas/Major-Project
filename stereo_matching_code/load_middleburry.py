import numpy as np
import os
from scipy import ndimage

def processs_input_image(image):
    image=np.array(image,dtype=np.float32)
    image=(image-np.mean(image))/np.std(image)
    return image


def load_middleburry():
    training_list_left = []
    training_list_right = []
    for i in range(0,200):
        if i in validation_index:
            validation_list_left.append(os.path.join(kitti_2015_path, "training/image_2",str(i).zfill(6)+"_10.png"))
            validation_list_right.append(os.path.join(kitti_2015_path, "training/image_3",str(i).zfill(6)+"_10.png"))
            validation_list_noc_label.append(os.path.join(kitti_2015_path, "training/disp_noc_0",str(i).zfill(6)+"_10.png"))
 
        else:
            training_list_left.append(os.path.join(kitti_2015_path, "training/image_2",str(i).zfill(6)+"_10.png"))
            training_list_right.append(os.path.join(kitti_2015_path, "training/image_3",str(i).zfill(6)+"_10.png"))
            training_list_noc_label.append(os.path.join(kitti_2015_path, "training/disp_noc_0",str(i).zfill(6)+"_10.png"))

    left_images,right_images,disp_images = load_all_images(training_list_left,training_list_right,training_list_noc_label)
    left_images_validation,right_images_validation,disp_images_validation = load_all_images(validation_list_left,validation_list_right,validation_list_noc_label)
       