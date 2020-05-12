import tensorflow as tf
from scipy import ndimage
import numpy as np
import cv2



L = 17
T = 20


class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (row, column) for cardinal direction.
        :param name: common name of said direction.
        """
        self.direction = direction
        self.name = name
        
N = Direction((-1, 0), name="Top")
S = Direction(( 1, 0), name="Down")
E = Direction(( 0, 1), name="Right")
W = Direction(( 0,-1), name="Left")


class CBCA:
    def __init__(self, max_length, limit):
        self.max_length = max_length
        self.limit = limit
        self.paths  = [W, E, N, S]
   
        
    def find_arms(self, image):
        arm_left   = np.zeros(shape = image.shape, dtype = 'int32')
        arm_right  = np.zeros(shape = image.shape, dtype = 'int32')
        arm_top    = np.zeros(shape = image.shape, dtype = 'int32')
        arm_bottom = np.zeros(shape = image.shape, dtype = 'int32')
        # self.running_num_vertical = np.zeros(shape = image.shape, dtype = 'int32')
        # self.total_num_region = np.zeros(shape = image.shape, dtype = 'int16')
        for path in self.paths:
            mask_new        = np.ones(shape = image.shape, dtype = 'int32')
            out_prev        = np.zeros (shape = image.shape, dtype = 'int32')

            if path.name == E.name:
                pad_image = image.copy()
                pad_image = np.pad(pad_image, pad_width =((0,0),(0,self.max_length)), constant_values=1000)
                
                for i in range(1, self.max_length+1): #for image width index
                    mask = np.abs(image[:,:] - pad_image[:,i:self.width+i]) <= self.limit
                    #print(mask)
                    mask_new = np.multiply(mask, mask_new)
                    #print(mask_new)
                    out  = i * mask_new
                    out  = np.maximum(out, out_prev)
                    out_prev = out
                arm_right = out
                
            if path.name == W.name:
                pad_image = image.copy()
                pad_image = np.pad(pad_image, pad_width =((0,0),(self.max_length,0)), constant_values=1000)
                pad_image_width = pad_image.shape[1]
                for i in range(1, self.max_length+1): #for image width index
                    mask = np.abs(self.image[:,:] - pad_image[:,pad_image_width-self.width-i:pad_image_width-i]) <= self.limit
                    #print(mask)
                    mask_new = np.multiply(mask, mask_new)
                    #print(mask_new)
                    out  = i * mask_new
                    out  = np.maximum(out, out_prev)
                    out_prev = out
                arm_left = out
                
                
            if path.name == S.name:
                pad_image = image.copy()
                pad_image = np.pad(pad_image, pad_width =((0, self.max_length),(0,0)), constant_values=1000)
                pad_image_width = pad_image.shape[1]
                for i in range(1, self.max_length): #for image width index
                    mask = np.abs(self.image[:,:] - pad_image[i:self.height+i,:]) <= self.limit
                    #print(mask)
                    mask_new = np.multiply(mask, mask_new)
                    #print(mask_new)
                    out  = i * mask_new
                    out  = np.maximum(out, out_prev)
                    out_prev = out
                arm_bottom = out
                
                
            if path.name == N.name:
                pad_image = image.copy()
                pad_image = np.pad(pad_image, pad_width =((self.max_length, 0),(0,0)), constant_values=1000)
                pad_image_height = pad_image.shape[0]
                for i in range(1, self.max_length+1): #for image width index
                    mask = np.abs(self.image[:,:] - pad_image[(pad_image_height-self.height-i):(pad_image_height-i) ,:]) <= self.limit
                    mask_new = np.multiply(mask, mask_new)
                    out  = i * mask_new
                    out  = np.maximum(out, out_prev)
                    out_prev = out
                arm_top = out
                
        return arm_right, arm_left, arm_bottom, arm_top
    
    def find_combined_arm(left_arm, right_arm):
        return np.minimum(left_arm, right_arm)
                
    def find_horizantal_sum(self):
        
        width = self.horizantal_running_sum.shape[1]
        self.horizantal_running_sum[:,0] = self.matching_cost[:,0]
        for i in range(1, width):
            self.horizantal_running_sum[:,i] = self.horizantal_running_sum[:,i-1] + self.matching_cost[:,i]
            
    def find_horizantal_integral(self):
        for y in range(self.height):
            for x in range(self.width):
                if x - self.arm_left[y,x]-1 < 0:
                    horizantal_sum = 0
                else:
                    horizantal_sum = self.horizantal_running_sum[y,x - self.arm_left[y,x]-1] 

                self.horizantal_integral[y,x] = self.horizantal_running_sum[y, x + self.arm_right[y,x]] - horizantal_sum
                
                
    def find_vertical_running_sum(self):
        height = self.vertical_running_sum.shape[0]
        self.vertical_running_sum[0,:] = self.horizantal_integral[0,:]
        # self.running_num_vertical[0,:] = self.num_horizantal_arm[0,:]
        for i in range(1, height):
            self.vertical_running_sum[i,:] = self.vertical_running_sum[i-1,:] + self.horizantal_integral[i,:]
            # self.running_num_vertical[i,:] = self.running_num_vertical[i-1,:] + self.num_horizantal_arm[i,:]
    def find_vertical_integral(self):
        for y in range(self.height):
           
            for x in range(self.width):
               
                if y - self.arm_top[y,x] -1 < 0:
                    vertical_sum = 0
                    # vertical_num = 0
                else:
                    vertical_sum = self.vertical_running_sum[y - self.arm_top[y,x] -1, x]
                    # vertical_num = self.running_num_vertical[y - self.arm_top[y,x] -1, x]
                    
                self.vertical_integral[y,x] = self.vertical_running_sum[y+self.arm_bottom[y,x],x] - vertical_sum
                # self.total_num_region[y,x] =  self.running_num_vertical[y+self.arm_bottom[y,x],x] - vertical_num
#                 num_supported_region = self.arm_left[y, x] + self.arm_right[y,x] + self.arm_bottom[y,x] + self.arm_top[y,x] + 1
        
                # self.vertical_integral[y,x] = self.vertical_integral[y,x].astype('float') / self.total_num_region[y,x].astype('float')
        
                
        
    def find_cbca(self,imageL, matching_cost):
        self.image = imageL
        self.matching_cost = matching_cost
        self.width = imageL.shape[1]
        self.height= imageL.shape[0]

        self.horizantal_running_sum = np.zeros_like(self.matching_cost)
        self.horizantal_integral = np.zeros_like(self.matching_cost)
        self.vertical_running_sum = np.zeros_like(self.matching_cost)
        self.vertical_integral = np.zeros_like(self.matching_cost)
    
        
        
        #step:1 finding arm_lengths in both right and left images
        L_arm_right, L_arm_left, L_arm_bottom, L_arm_top = self.find_arms(imageL) 
        self.arm_right, self.arm_left, self.arm_bottom, self.arm_top = L_arm_right, L_arm_left, L_arm_bottom, L_arm_top

        # self.num_horizantal_arm = np.add(self.arm_right, self.arm_left) + 1
        #step:2 Finding intersection length in both right and left images
#         LR_arm_right = find_combined_arm(L_arm_right, R_arm_right)
#         LR_arm_left = find_combined_arm(L_arm_left, R_arm_left)
#         LR_arm_bottom = find_combined_arm(L_arm_bottom, R_arm_bottom)
#         LR_arm_top = find_combined_arm(L_arm_top, R_arm_top)
        
        #step:3 Finding horizantal running sum (Cummulative sum) of matching cost
        self.find_horizantal_sum()
        
        #step:4 Finding horizantal integral (Horizantal aggregation step)
        self.find_horizantal_integral()
        
        #step:5 Finding vertical running sum (Cummulative sum) of horizantal integral
        self.find_vertical_running_sum()
        
        #step:6 Finding vertical integral (complete aggregate)
        self.find_vertical_integral()
        
        
        return self.vertical_integral
        

