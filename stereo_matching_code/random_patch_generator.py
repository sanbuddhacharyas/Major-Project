def disp_image_to_label(disp_image,nclasses):
    disp_image[disp_image>nclasses-1]=0
    return disp_image

def get_valid_pixels(training_list_noc_label,total_patch_size,maxDisp):
    half_path_size=int(total_patch_size/2)
    half_max_disp=int(maxDisp/2)
    list_of_valid_pixels=[]
    for i in range(len(training_list_noc_label)):
        valid_choices=np.where(training_list_noc_label[i]!=0)
        valid_choices=list(map(list, zip(*valid_choices)))
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]>half_path_size]
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]<training_list_noc_label[i].shape[0]-half_path_size]
        #
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]>half_path_size+maxDisp+training_list_noc_label[i][x_y_pair[0]][x_y_pair[1]]]
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]<training_list_noc_label[i].shape[1]-half_max_disp-half_path_size]
        list_of_valid_pixels.append(valid_choices)
    return list_of_valid_pixels

def load_random_patch(training_list_left,training_list_right,training_list_noc_label,receptive_field,receptive_field_right,maxDisp,batch_size,
valid_pixels_list):
    total_patch_size=receptive_field
    half_path_size=int(total_patch_size/2)
    half_path_size_w=int(receptive_field_right/2)

    batch_left=np.zeros((0, total_patch_size,receptive_field_right,3), dtype=np.float32)
    batch_right=np.zeros((0, total_patch_size,maxDisp+receptive_field_right,3), dtype=np.float32)
    batch_disp=np.zeros((0, total_patch_size,receptive_field_right), dtype=np.float32)

    for batch in range(0,batch_size):
        random_image_index=np.random.randint(len(training_list_left))        
        random_choice=np.random.randint(len(valid_pixels_list[random_image_index]))
        x_rand,y_rand=valid_pixels_list[random_image_index][random_choice][0],valid_pixels_list[random_image_index][random_choice][1]
        left=training_list_left[random_image_index][x_rand-half_path_size:x_rand+half_path_size,
        y_rand-half_path_size_w:y_rand+half_path_size_w,:]
        d=training_list_noc_label[random_image_index][x_rand-half_path_size:x_rand+half_path_size,y_rand-half_path_size:y_rand+half_path_size]
        right=training_list_right[random_image_index][x_rand-half_path_size:x_rand+half_path_size,y_rand-half_path_size-maxDisp:y_rand+half_path_size,:]


        d=disp_image_to_label(d,maxDisp+1)

        left = left[np.newaxis,...]
        right = right[np.newaxis,...]
        d= d[np.newaxis,...]

        batch_left=np.concatenate([batch_left, left],axis=0)
        batch_right=np.concatenate([batch_right, right],axis=0)
        batch_disp=np.concatenate([batch_disp, d],axis=0)

    return batch_left,batch_right,batch_disp

def load_kitti():
    print('load_images:')
    left_images = np.load('tensorflow_workstation/anita/left_images_1.npy')
    right_images = np.load('tensorflow_workstation/anita/right_images_1.npy')
    disp_images = np.load('tensorflow_workstation/anita/disp_images_1.npy')
    left_images_validation = np.load('tensorflow_workstation/anita/left_images_validation_1.npy')
    right_images_validation = np.load('tensorflow_workstation/anita/right_images_validation_1.npy')
    disp_images_validation = np.load('tensorflow_workstation/anita/disp_images_validation_1.npy')
    
    print('process images:')
    valid_pixels_train = get_valid_pixels(disp_images,receptive_field,maxDisp)
    valid_pixels_val = get_valid_pixels(disp_images_validation,receptive_field,maxDisp)
    #np.save('valid_pixels_train_1',valid_pixels_train)
    #np.save('valid_pixels_val_1',valid_pixels_val)
    #valid_pixels_train  = np.load('tensorflow_workstation/anita/valid_pixels_train.npy')
    #valid_pixels_val = np.load('tensorflow_workstation/anita/valid_pixels_val.npy')
    print('Valid pixels extracted!')
    return left_images, right_images, disp_images, left_images_validation, right_images_validation, disp_images_validation, valid_pixels_train, valid_pixels_val

