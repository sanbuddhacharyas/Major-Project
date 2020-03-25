import numpy as np
import cv2
import glob
import argparse
import h5py


class StereoCalibration(object):
    def __init__(self, filepath):
        
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6*8, 3), np.float32)
     
        self.objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
        self.objp = 30 * self.objp

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.num_detected_board = 0
        #self.read_images(self.cal_path)
        pattern = (8,6)

    def calibrate_camera(self, write):
        pattern = (8,6)
     
        number_cal_pair = np.array(glob.glob(self.cal_path+'/Stereo_images/Right/*.jpg')).size
        
        for i in range(number_cal_pair):

            img_l = cv2.imread(self.cal_path+'/Stereo_images/Left/'+str(i)+'.jpg')
            img_r = cv2.imread(self.cal_path+'/Stereo_images/Right/'+str(i)+'.jpg')

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r,  pattern, None)

           

            if ret_l is True and ret_r is True:

                #If corner is detected with different order then rotate detected points to match with object 
                # points order(0,0,0), (1,0,0)...
                if (corners_l[0][0][0] > corners_l[47][0][0]):
                    corners_l = np.rot90(corners_l,2).reshape(48,1,2)
                    corners_l = np.array(corners_l)
                
                if (corners_r[0][0][0] > corners_r[47][0][0]):
                    corners_r = np.rot90(corners_r,2).reshape(48,1,2)
                    corners_r = np.array(corners_r)

                #Find subPixel of left camera and add image points
                cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                
                #Find subPixel of Right camera and add image points
                cv2.cornerSubPix(gray_r, corners_r, (11, 11),(-1, -1), self.criteria)
                

                # Draw and display the corners Left image
                ret_l = cv2.drawChessboardCorners(img_l, pattern,corners_l, ret_l)
                cv2.imshow("Left"+str(i), img_l)

                # Draw and display the corners Right image
                ret_r = cv2.drawChessboardCorners(img_r, pattern ,corners_r, ret_r)
                cv2.imshow("Right"+str(i), img_r)

                #Wait until any key is pressed
                key = cv2.waitKey()
                if key == 115:
                    self.imgpoints_l.append(corners_l)
                    self.imgpoints_r.append(corners_r)

                    #Add object points(Object World point)
                    self.objpoints.append(self.objp)

                    #Num of pair chess board_pair detected
                    self.num_detected_board += 1

                cv2.destroyAllWindows()

           
        #Transpose image shape
        self.img_shape = gray_l.shape[::-1]
        #assert(self.num_detected_board == np.array(self.imgpoints_l).size)
        print("Numbers of Chess detected" , self.num_detected_board, "Img _points", np.array(self.imgpoints_l).shape)
        flags = 0
        flags |= 8
        #Determine the camera parameters
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None, flags = flags)

        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None, flags = flags)

        camera_parameters = {   "M1": self.M1,
                                "dist1": self.d1,
                                "r1": self.r1,
                                "t1":self.t1,
                                "M2":self.M2,
                                "dist2": self.d2,
                                "r2":self.r2,
                                "t2":self.t2
                            }

        if write == True:
            with h5py.File(self.cal_path+"/Calibration1/Camera_parameters.h5py","w") as hdf:
                Camera_p = hdf.create_group("Camera_parameter")
                Camera_p.create_dataset("M1",data = camera_parameters["M1"])
                Camera_p.create_dataset("dist1",data = camera_parameters["dist1"] )
                Camera_p.create_dataset("M2",data = camera_parameters["M2"] )
                Camera_p.create_dataset("dist2",data = camera_parameters["dist2"] )
                Camera_p.create_dataset("t1",data = camera_parameters["t1"] )
                Camera_p.create_dataset("t2",data = camera_parameters["t2"] )
                Camera_p.create_dataset("r1",data = camera_parameters["r1"] )
                Camera_p.create_dataset("r2",data = camera_parameters["r2"] )
                
        return camera_parameters
       
                                
        

    def stereo_calibrate(self , write):
        self.calibrate_camera(True)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        
        flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH |
                cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |
                cv2.CALIB_FIX_K6)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, self.img_shape,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

    
        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()

        if write == True:
            with h5py.File(self.cal_path+"/Calibration1/Stereo_parameters.h5py","w") as hdf:
                #Creating group name camera_parameters
                Camera_p = hdf.create_group("Stereo_parameters")
                #Creating data set in group
                Camera_p.create_dataset("M1",data = camera_model["M1"])
                Camera_p.create_dataset("dist1",data = camera_model["dist1"] )
                Camera_p.create_dataset("M2",data = camera_model["M2"] )
                Camera_p.create_dataset("dist2",data = camera_model["dist2"] )
                Camera_p.create_dataset("R",data = camera_model["R"] )
                Camera_p.create_dataset("T",data = camera_model["T"] )
                Camera_p.create_dataset("E",data = camera_model["E"] )
                Camera_p.create_dataset("F",data = camera_model["F"] )
            
        return camera_model
    
    def stereo_rectify(self):
        camera_model = self.stereo_calibrate(True)
        #perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_model["M1"], camera_model["dist1"],
                                                        camera_model["M2"], camera_model["dist2"],
                                                        self.img_shape,
                                                        camera_model["R"], camera_model["T"], alpha=1)

        with h5py.File(self.cal_path+"/Calibration1/Stereo_rectify.h5py","w") as hdf:
            #Creating group name camera_parameters
            stereo_rectify = hdf.create_group("Stereo_rectify")
            #Creating data set in group
            stereo_rectify.create_dataset("R1",data = R1)
            stereo_rectify.create_dataset("R2",data = R2 )
            stereo_rectify.create_dataset("P1",data = P1 )
            stereo_rectify.create_dataset("P2",data = P2 )
            stereo_rectify.create_dataset("Q",data =  Q )
            stereo_rectify.create_dataset("roi1",data = roi1 )
            stereo_rectify.create_dataset("roi2",data = roi2 )

        newcameramtx_1, roi_1 = cv2.getOptimalNewCameraMatrix(camera_model["M1"], camera_model["dist1"], 
                                                                self.img_shape, 1, self.img_shape)
        newcameramtx_2, roi_2 = cv2.getOptimalNewCameraMatrix(camera_model["M2"], camera_model["dist2"], 
                                                                self.img_shape, 1, self.img_shape)

        mapx1, mapy1 = cv2.initUndistortRectifyMap(camera_model["M1"], camera_model["dist1"], 
                                                    R1, P1, self.img_shape, cv2.CV_32F)

        mapx2, mapy2 = cv2.initUndistortRectifyMap(camera_model["M2"], camera_model["dist2"], R2, P2,
                                                   self.img_shape, cv2.CV_32F)


        with h5py.File(self.cal_path+"/Calibration1/stereo_mapping.h5py","w") as hdf:
            stereo_map = hdf.create_group("stereo_mapping")
            stereo_map.create_dataset("mapx1",data= mapx1)
            stereo_map.create_dataset("mapy1",data= mapy1)
            stereo_map.create_dataset("mapx2",data= mapx2)
            stereo_map.create_dataset("mapy2",data= mapy2)
            stereo_map.create_dataset("roi1",data= roi1)
            stereo_map.create_dataset("roi2",data= roi2)
        print("Calibrated sucessfull")

        img_l = cv2.imread(self.cal_path+'/Stereo_images/Left/'+str(0)+'.jpg')
        img_r = cv2.imread(self.cal_path+'/Stereo_images/Right/'+str(0)+'.jpg')
        img_rect1 = cv2.remap(img_l, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(img_r, mapx2, mapy2, cv2.INTER_LINEAR)
        print(roi2)
        # x1,y1,w1,h1 = roi1
        x2,y2,w2,h2 = roi2
        # img_rect1 = img_rect1[y1:y1+h1, x1:x1+w1]
        # img_rect2 = img_rect2[y2:y2+h2, x2:x2+w2]
        # dim = (1280, 720)

        # L_resized = cv2.resize(img_rect1, dim, interpolation = cv2.INTER_AREA)
        # R_resized = cv2.resize(img_rect2, dim, interpolation = cv2.INTER_AREA)


        img = cv2.hconcat([img_rect1, img_rect2])
        thickness = 2
        color = (0, 255, 0) 
        y = 0
        for i in range(15):
            start_point = (0, y)
            end_point = (2560, y) 
            y += 50
            
            image = cv2.line(img, start_point, end_point, color, thickness) 

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image",2560, 720)
        cv2.imshow("image",img)
        cv2.waitKey()
        cv2.destroyAllWindows()
            
   # def stereo_undistor(self):
        # img_l = cv2.imread(self.cal_path+'/Stereo_images/Left/'+str(1)+'.jpg')
        # gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        # img_shape = gray_l.shape[::-1]
        # with h5py.File(self.cal_path+"/Calibration/Camera_parameters.h5py","r") as hdf:
        #     print( hdf.keys())
        #     M1 = np.array(hdf["Camera_parameter/M1"])
        #     dist1 = np.array(hdf["Camera_parameter/dist1"])
        #     M2 = np.array(hdf["Camera_parameter/M2"])
        #     dist2 = np.array(hdf["Camera_parameter/dist2"])
        
        # with h5py.File(self.cal_path+"/Calibration/Stereo_rectify.h5py","r") as hdf:
        #     print(hdf.keys())
        #     R1 = np.array(hdf["Stereo_rectify/R1"])
        #     R2 = np.array(hdf["Stereo_rectify/R2"])
       
        

    def write_data(self,Calibration_parameter, Stereo_parameters ,map):
        with h5py.File("Calibration.h5py","w") as hdf:
            Camera_p = hdf.create_group("Camera_parameter")
            Stereo_p = hdf.create_group("Stereo_parameters")
            map_p = hdf.create_group("map")
            
            Camera_p.create_dataset("M1",data = Calibration_parameter["M1"])
            Camera_p.create_dataset("dist1",data = Calibration_parameter["dist1"] )
            Camera_p.create_dataset("M2",data = Calibration_parameter["M2"] )
            Camera_p.create_dataset("dist2",data = Calibration_parameter["dist2"] )
            Camera_p.create_dataset("R",data = Calibration_parameter["R"] )
            Camera_p.create_dataset("T",data = Calibration_parameter["T"] )
            Camera_p.create_dataset("E",data = Calibration_parameter["E"] )
            Camera_p.create_dataset("F",data = Calibration_parameter["F"] )

            Stereo_p.create_dataset("R1",data = Stereo_parameters["R1"])
            Stereo_p.create_dataset("R2",data = Stereo_parameters["R2"] )
            Stereo_p.create_dataset("P1",data = Stereo_parameters["P1"] )
            Stereo_p.create_dataset("P2",data = Stereo_parameters["P2"] )
            Stereo_p.create_dataset("Q",data = Stereo_parameters["Q"] )
            Stereo_p.create_dataset("roi1",data = Stereo_parameters["roi1"] )
            Stereo_p.create_dataset("roi2",data = Stereo_parameters["roi2"] )

            map_p.create_dataset("map1x",data=map["map1x"])
            map_p.create_dataset("map1y",data=map["map1y"])
            map_p.create_dataset("map2x",data=map["map2x"])
            map_p.create_dataset("map2y",data=map["map2y"])

           

            #data.create_dataset("Stereo_parameters",data= Stereo_parameters)

            

            

    def read_data(self):
        with h5py.File("Calibration.h5py","r") as hdf:
            hdf.keys()
            d ={
            "M1":np.array(hdf["Camera_parameter/M1"]),
            "dist1": np.array(hdf["Camera_parameter/dist1"]),
            "M2": np.array(hdf["Camera_parameter/M2"]),
            "dist2": np.array(hdf["Camera_parameter/dist2"]),
            "R" : np.array(hdf["Camera_parameter/R"]),
            "T" : np.array(hdf["Camera_parameter/T"]),
            "E" : np.array(hdf["Camera_parameter/E"]),
            "F" : np.array(hdf["Camera_parameter/F"])
            }
            stereo={
            "R1": np.array(hdf["Stereo_parameters/R1"]),
            "R2" : np.array(hdf["Stereo_parameters/R2"]),
            "P1" : np.array(hdf["Stereo_parameters/P1"]),
            "P2" : np.array(hdf["Stereo_parameters/P2"]),
            "Q" :  np.array(hdf["Stereo_parameters/Q"]),
            "roi1":np.array(hdf["Stereo_parameters/roi1"]),
            "roi2":np.array(hdf["Stereo_parameters/roi2"])
            }
        
            map ={
            "map1x":np.array(hdf["map/map1x"]),
            "map1y":np.array(hdf["map/map1y"]),
            "map2x":np.array(hdf["map/map2x"]),
            "map2y":np.array(hdf["map/map2y"])
            }
        
        return d, stereo,map
         


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
