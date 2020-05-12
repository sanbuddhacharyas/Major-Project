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

     
                
        return np.stack((arm_right, arm_left, arm_bottom, arm_top))
    
   
                
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
        

    def find_combined_arms(self):
        shape = (4, self.matching_cost.shape[0],self.matching_cost.shape[1])
        arm  = np.zeros(shape = shape, dtype = 'int32')
      
        armLeft = self.find_arms(self.imageL)
        armRight = self.find_arms(self.imageR)
        print("Left_img / Right_arm",armLeft[0])
        print("Right_img / Right_arm",armRight[0])
      
        # width = self.imageR.shape[1]
        d = np.random.randint(0, 5, size=(self.matching_cost.shape[0],self.matching_cost.shape[1]))
        print("D",d)
        for y in range(self.height):
            for x in range(self.width):
                if x-d[y,x] >= 0 :
                    arm[:,y,x] = np.minimum(armLeft[:,y,x], armRight[:,y,x-d[y,x]] )

                else:
                    arm[:,y,x] = armLeft[:,y,x]
                        
        return arm
        
    def find_cbca(self,imageL, imageR, matching_cost):
        self.imageL = imageL
        self.imageR = imageR
        self.matching_cost = matching_cost
        self.width = imageL.shape[1]
        self.height= imageL.shape[0]

        self.horizantal_running_sum = np.zeros_like(self.matching_cost)
        self.horizantal_integral = np.zeros_like(self.matching_cost)
        self.vertical_running_sum = np.zeros_like(self.matching_cost)
        self.vertical_integral = np.zeros_like(self.matching_cost)
    
        
        
        #step:1 finding arm_lengths in both right and left images
       
        arm = self.find_combined_arms()
        print("arm",arm[0])
        self.arm_right, self.arm_left, self.arm_bottom, self.arm_top = arm[0] , arm[1] ,arm[2], arm[3]

        
        #step:3 Finding horizantal running sum (Cummulative sum) of matching cost
        self.find_horizantal_sum()
        
        #step:4 Finding horizantal integral (Horizantal aggregation step)
        self.find_horizantal_integral()
        
        #step:5 Finding vertical running sum (Cummulative sum) of horizantal integral
        self.find_vertical_running_sum()
        
        #step:6 Finding vertical integral (complete aggregate)
        self.find_vertical_integral()
        
        
        return self.vertical_integral
        

# if __name__ == '__main__':
#     num_classes = 129
#     imgL = np.array([[20, 21, 20, 50, 200,  20, 20, 20],
#                     [21, 22, 0,  50,  50,  50, 51,  0],
#                     [23,  0, 50, 51, 100, 100,  0,  0],
#                     [0,  55, 55, 50,  60, 200, 100, 100],
#                     [50, 50, 50, 50,  50,  50, 50,  50]])
    
#     imgR = np.array([[0,20, 21, 20, 50, 200,  20, 20 ],
#                     [10,21, 22, 0,  50,  50,  50, 51  ],
#                     [20,23,  0, 50, 51, 100, 100,  0  ],
#                     [20,30,0,  55, 55, 50,  60, 200,  ],
#                     [55,40,50, 50, 50, 50,  50,  50, ]])
    

#     matching_cost_shape = (imgL.shape[0], imgL.shape[1], num_classes)
#     np.random.seed(0)
#     matching_cost = np.random.randint(0, 20,size=(matching_cost_shape))
    
    
#     cost_agg =CBCA(8, T)
#     out = cost_agg.find_cbca(imgL,imgR, matching_cost)
# #     #cost_agg.find_arms()
# #     print("Left")
# #     print(cost_agg.arm_left[:,:,0])
# #     print("Right")
# #     print(cost_agg.arm_right[:,:,0])
# #    # print("Running_sum",cost_agg.horizantal_running_sum)


# #     print("matching_cost")
# #     print(matching_cost[:,:,0])
# #     print("Vertical_integral")
# #     print( cost_agg.vertical_integral[:,:,0])
 

