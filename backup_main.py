import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calib, undistort
from backup_thresh import gradient_combine, hls_combine, comb_result
from backup_lane import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map
import socket
import time
from cam_util import get_img
from lane_util import  binary_pipeline, fit_track_lanes,callback_ths, visualize_images

params_cam = {
    "WIDTH": 640, # image width
    "HEIGHT": 360, # image height
    "FOV": 90, # Field of view
    "localIP": "127.0.0.1",
    "localPort": 1232,
    "Block_SIZE": int(65000),
    "X": 0, # meter
    "Y": 0,
    "Z": 1.0,
    "YAW": 0, # deg
    "PITCH": 0,
    "ROLL": 0
}
UDP_cam = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDP_cam.bind((params_cam["localIP"], params_cam["localPort"]))

left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()
####################################################################################################
if __name__ == '__main__':

    while (1):
        frame = get_img(UDP_cam, params_cam)
        #frame = cv2.resize(frame, (480, 320))

        # Correcting for Distortion
        undist_img = undistort(frame, mtx, dist)
        # resize video #영역 보간법(축소시)
        #undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        rows, cols = undist_img.shape[:2]   #(height × width × other dimensions)
        
        #sobel operation
        
        combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
        cv2.imshow('gradient combined image', combined_gradient)
        combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
        cv2.imshow('HLS combined image', combined_hls)


        combined_result = comb_result(combined_gradient, combined_hls)
        cv2.imshow('combined image', combined_result)
        #c_rows = 118,c_cols = 640
        
        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [c_cols / 2 - 24, 10], [c_cols / 2 + 24, 10]
        s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows] 
        
        src = np.float64([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float64([(170, 720), (170, 0), (550, 0), (550, 720)])

        warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
        cv2.imshow('warp', warp_img)
        #searching_img = find_LR_lines(warp_img, left_line, right_line)                      left_line, right_line = line()
        searching_img = find_LR_lines(warp_img, left_line, right_line)
        cv2.imshow('LR searching', searching_img)

        w_comb_result, w_color_result = draw_lane(searching_img, left_line,right_line)     #right_line 생략
        cv2.imshow('w_comb_result', w_comb_result)
        #cv2.imshow('w_color_result', w_color_result)
        #여기까지 bird eye view


        # Drawing the lines back down onto the road
        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))          #역 변환
        lane_color = np.zeros_like(undist_img)
        lane_color[170:rows - 12, 0:cols] = color_result    #12 to 22#####################################################
        #cv2.imshow('color_result', color_result)#

        # Combine the result with the original image                                #색에 0.8가중치, 그라디언트에 0.2 가중치 -> cracking이나 까만 이상한선 제거
        result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
        cv2.imshow('result', result.astype(np.uint8))

        info, info2 = np.zeros_like(result),  np.zeros_like(result)                 #road info, 위치 이미지의 자리 마련
        info[5:110, 5:190] = (255, 255, 255)                                        #road status
        info2[5:110, cols-111:cols-6] = (255, 255, 255)                             #road map
            
        info = cv2.addWeighted(result, 1, info, 0.2, 0)
        info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)

        road_map = print_road_map(w_color_result, left_line, right_line)
        info2[10:105, cols-106:cols-11] = road_map
       
        info2 = print_road_status(info2, left_line, right_line)     
        cv2.imshow('road info', info2)

        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.waitKey(0)
        #if cv2.waitKey(1) & 0xFF == ord('r'):
        #    cv2.imwrite('check1.jpg', undist_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    UDP_cam.close()
    #cap.release()
    cv2.destroyAllWindows()
