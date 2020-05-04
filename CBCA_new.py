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
                    mask = np.abs(image[:,:] - pad_image[:,pad_image_width-self.width-i:pad_image_width-i]) <= self.limit
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
                    mask = np.abs(image[:,:] - pad_image[i:self.height+i,:]) <= self.limit
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
                    mask = np.abs(image[:,:] - pad_image[(pad_image_height-self.height-i):(pad_image_height-i) ,:]) <= self.limit
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
        for d in range(self.num_classes):
            for y in range(self.height):
                for x in range(self.width):
                    if x - self.arm_left[y,x,d]-1 < 0:
                        horizantal_sum = 0
                    else:
                        horizantal_sum = self.horizantal_running_sum[y,x - self.arm_left[y,x,d]-1,d]
                        
                    if x + self.arm_right[y,x,d] > (self.width - 1):
                        w = self.width -1
                    else:
                        w = x + self.arm_right[y,x,d]

                    self.horizantal_integral[y,x,d] = self.horizantal_running_sum[y, w ,d] - horizantal_sum

                
    def find_vertical_running_sum(self):
        height = self.vertical_running_sum.shape[0]
        self.vertical_running_sum[0,:] = self.horizantal_integral[0,:]
    
        for i in range(1, height):
            self.vertical_running_sum[i,:] = self.vertical_running_sum[i-1,:] + self.horizantal_integral[i,:]
           
    def find_vertical_integral(self):
        for d in range(self.num_classes):
            for y in range(self.height):

                for x in range(self.width):

                    if y - self.arm_top[y,x,d] -1 < 0:
                        vertical_sum = 0
                    else:
                        vertical_sum = self.vertical_running_sum[y - self.arm_top[y,x,d] -1, x,d]
                       

                    self.vertical_integral[y,x,d] = self.vertical_running_sum[y+self.arm_bottom[y,x,d],x,d] - vertical_sum

                
    
    def find_combined_arms(self):
        self.arm_left   = np.zeros(shape = self.matching_cost.shape, dtype = 'int32')
        self.arm_right  = np.zeros(shape = self.matching_cost.shape, dtype = 'int32')
        self.arm_top    = np.zeros(shape = self.matching_cost.shape, dtype = 'int32')
        self.arm_bottom = np.zeros(shape = self.matching_cost.shape, dtype = 'int32')
        L_arm_right, L_arm_left, L_arm_bottom, L_arm_top = self.find_arms(self.imageL)
        R_arm_right, R_arm_left, R_arm_bottom, R_arm_top = self.find_arms(self.imageR)
        width = self.imageR.shape[1]
        for i in range(num_classes):
            self.arm_right[:,:,i] = np.pad(R_arm_right[:,:width-i], pad_width =((0, 0),(i,0)), constant_values=0)
            self.arm_left[:,:,i]  = np.pad(R_arm_left[:,:width-i], pad_width =((0, 0),(i,0)), constant_values=0)
            self.arm_top[:,:,i]  =  np.pad(R_arm_top[:,:width-i], pad_width =((0, 0),(i,0)), constant_values=0)
            self.arm_bottom[:,:,i]= np.pad(R_arm_bottom[:,:width-i], pad_width =((0, 0),(i,0)), constant_values=0)
            
            #find union of both left and right region ie.max
            self.arm_right[:,:,i] = np.maximum(L_arm_right, self.arm_right[:,:,i] )
            self.arm_left[:,:,i] =  np.maximum(L_arm_left, self.arm_left[:,:,i] )
            self.arm_top[:,:,i] =   np.maximum(L_arm_top, self.arm_top[:,:,i] )
            self.arm_bottom[:,:,i] = np.maximum(L_arm_bottom, self.arm_bottom[:,:,i] )
                
        
    def find_cbca(self,imageL, imageR, matching_cost,num_classes):
        self.imageL = imageL
        self.imageR = imageR
        self.matching_cost = matching_cost
        self.width = imageL.shape[1]
        self.height= imageL.shape[0]
        self.num_classes = num_classes

        self.horizantal_running_sum = np.zeros_like(self.matching_cost)
        self.horizantal_integral = np.zeros_like(self.matching_cost)
        self.vertical_running_sum = np.zeros_like(self.matching_cost)
        self.vertical_integral = np.zeros_like(self.matching_cost)
    
        
        
        #step:1 finding arm_lengths in both right and left images
        
        self.find_combined_arms()
        
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
        