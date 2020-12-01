import numpy as np
import cv2
from PIL import Image
import matplotlib.image as mpimg
import sys
#global left_line_length = 0                # 탐지조건을 위한 거리 (표준편차 대신)
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Set the width of the windows +/- margin
        self.window_margin = 56
        # x values of the fitted line over the last n iterations
        self.prevx = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        # starting x_value
        self.startx = None
        # ending x_value
        self.endx = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # road information
        self.road_inf = None
        self.curvature = None
        self.deviation = None

def warp_image(img, src, dst, size):
    """ Perspective Transform """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv

def rad_of_curvature(left_line,right_line):        # right_line 삭제
    """ measure radius of curvature  """

    ploty = left_line.ally
    leftx, rightx = left_line.allx, right_line.allx    #, rightx = , right_line.allx 삭제

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Define conversions in x and y from pixels space to meters
    width_lanes = abs(right_line.startx - left_line.startx)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7*(720/1280) / width_lanes  # meters per pixel in x dimension                 / 차선폭

    # Define y-value where we want radius of curvature
    # the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)                       #  x,y축 -> a b c
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature                                                   #포물선 곡률반지름 구하는 공식
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(      #y_eval * ym_per_pix  : m로 환산 ->30픽셀 이 1m
        2 * left_fit_cr[0])                                           ##(2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) -> 기울기
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # radius of curvature result
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad

def smoothing(lines, pre_lines=3):                      #선을 수식화 할수 있도록 일자로 나타내는것
    # collect lines & print average line
    lines = np.squeeze(lines)                               #squeeze : []를 한꺼풀 벗기는거 
    avg_line = np.zeros((720))                              #0으로 채워진  720개의 1차원 배열 생성=>초기화

    for ii, line in enumerate(reversed(lines)):             #배열을 뒤집어서
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line
    
left_line_length=0

def blind_search(b_img, left_line,right_line): # right_line주석
    """
    blind search - first frame, lost lane lines
    using histogram & sliding window
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis=0)           #누적합, (b_img.shape=>720,720)--->>>[(360,360); , ;]
                                                                             #아래쪽 4분의 1의 픽셀 다더해(좌측 or 우측 차선and 알파 픽셀값들)
    
    # Create an output image to draw on and  visualize the result
    output = np.dstack((b_img, b_img, b_img)) * 255                         # b_img 3채널짜리 새하얀색(모든픽세255) 이미지 창조(도화지?)  ->  (360,360,3)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)                                #360
    start_leftX = np.argmax(histogram[:midpoint])                            #함수가 최대값인 x의값들 <시작부터 중간까지중> start_leftX ->ex 184
    #print(start_leftX)
    start_rightX = np.argmax(histogram[midpoint:]) + midpoint                #함수가 최대값인 x의값들 <중간부터 끝까지중> start_rightX + midpoint  ->ex 614
    #print(start_rightX)


    #좌우차선을 hist가 가장큰 값으로 정한다.


    
    # Choose the number of sliding windows
    num_windows = 9
    # Set height of windows
    window_height = np.int(b_img.shape[0] / num_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()           #차선인덱스들의 2차원 어레이
    nonzeroy = np.array(nonzero[0])     #아래로길이
    nonzerox = np.array(nonzero[1])     #옆으로길이

    # Current positions to be updated for each window
    current_leftX = start_leftX                                                 #윈도우 만들려고 현재값을 저장.
    current_rightX = start_rightX

    # Set minimum number of pixels found to recenter window
    min_num_pixel = 50

    # Create empty lists to receive left and right lane pixel indices
    win_left_lane = []
    win_right_lane = []

    window_margin = left_line.window_margin

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = b_img.shape[0] - (window + 1) * window_height         #아래점      #9칸짜리 사각형 만들기 높이를 9로 나누어서 사각형에 인자로 넣어줄 아래 위 값을 찾음
        win_y_high = b_img.shape[0] - window * window_height              #위점           #window 크기
        win_leftx_min = current_leftX - window_margin                           #window 폭 (왼차선)
        win_leftx_max = current_leftX + window_margin
        win_rightx_min = current_rightX - window_margin                         #window 폭 (오른차선)
        win_rightx_max = current_rightX + window_margin

        # Draw the windows on the visualization image
        cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)      #win_y_low는 윈도우중 맨아래 윈도우인데 낮은 아랫면의 값
        cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (nonzerox <= win_leftx_max)).nonzero()[0]

        #print(left_window_inds)

        #print(left_window_inds)
        #메모장출력                                                                            #nonzero인 픽셀의 갯수 다합친건가???????
        #sys.stdout = open('output.txt','w')                                                  #[12516 12517 12518 ... 24365 24366 24367]

        right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
        nonzerox <= win_rightx_max)).nonzero()[0]
        # Append these indices to the lists
        win_left_lane.append(left_window_inds)    
        win_right_lane.append(right_window_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(left_window_inds) > min_num_pixel:       #아래로 인덱스수가 좀 크면 평균인 인덱스를 중심으로  
            current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_rightX = np.int(np.mean(nonzerox[right_window_inds]))
    
    # Concatenate the arrays of indices
    win_left_lane = np.concatenate(win_left_lane)           ##윈도우 서칭한 결과를 합친다...
    win_right_lane = np.concatenate(win_right_lane)

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]                 #이게 왼차선의 픽셀위치(인덱스)
    rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]             #이게 우차선의 픽셀위치

    output[lefty, leftx] = [255, 0, 0]                                              #차선위치를 OUTPUT에 저장해서 리턴할것
    output[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)                                          #np.polyfit ==>>다항함수로 만들기, 2차식 ->[A B C]리턴 = Ax^2+Bx+C
    right_fit = np.polyfit(righty, rightx, 2)                                       #차선을 다항함수로 표현

    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])                      #np.linspace(시작,끝,갯수), 시작과 끝을 균일하게 나누는 점을 생성해줌 근데갯수가 픽셀값=>모든픽셀

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]       #left_plotx=> 왼차선의 함수(배열)를 나타내는것 ->ay^2 + by + c
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_line.prevx.append(left_plotx)                                              #함수 :  점의 집합.
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10: 
        #print(left_line.prevx)
        left_avg_line = smoothing(left_line.prevx, 10)
        ##print(left_avg_line)#################################################################3
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        ###print(left_fit)###############################################################           a,b,c값
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    
    
    left_line.startx= left_line.allx[len(left_line.allx)-1]
    right_line.startx =  right_line.allx[len(right_line.allx)-1]

    left_line.endx = left_line.allx[0]    #right_line.endx right_line.allx[0]

    left_line.detected= True
    right_line.detected = True                #여기까지 차선인식
    # print radius of curvature
    rad_of_curvature(left_line,right_line)     # right_line 삭제
    return output






def prev_window_refer(b_img, left_line,right_line):    #right_line삭제
    """
    refer to previous window info - after detecting lane lines in previous frame
    """
    # Create an output image to draw on and  visualize the result
    output = np.dstack((b_img, b_img, b_img)) * 255                      #결과 : 일단 전체그림

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Set margin of windows
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # Identify the nonzero pixels in x and y within the window
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    #cv2.imshow('left_inds',left_inds)
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    output[lefty, leftx] = [255, 0, 0]              #빨
    output[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    leftx_avg = np.average(left_plotx)
    rightx_avg = np.average(right_plotx)

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty
    
    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty
    
    # goto blind_search if the standard value of lane lines is high.
    standard = np.std(right_line.allx - left_line.allx)         

    if (standard > 80):
        left_line.detected = False

    left_line.startx = left_line.allx[len(left_line.allx) - 1]
    right_line.startx = right_line.allx[len(right_line.allx) - 1]
    left_line.endx = left_line.allx[0]
    right_line.endx = right_line.allx[0]

    # print radius of curvature
    rad_of_curvature(left_line, right_line)     
    return output

def find_LR_lines(binary_img, left_line, right_line):    
    """
    find left, right lines & isolate left, right lines
    blind search - first frame, lost lane lines
    previous window - after detecting lane lines in previous frame
    """

    # if don't have lane lines info
    if left_line.detected == False:                                 
        return blind_search(binary_img, left_line, right_line)   
    # if have lane lines info
    else:
        return prev_window_refer(binary_img, left_line,right_line)     

def draw_lane(img, left_line, right_line, lane_color=(255, 0, 255), road_color=(0, 255, 0)):        #, right_line(2번째 인자 삭제)
    """ draw lane lines & current driving space """ #보라색 녹색
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx   #, right_plotx = , right_line.allx
    ploty = left_line.ally

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_pts_l = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
    right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

    return result, window_img

def road_info(left_line, right_line):        
    """ print road information onto result image """
    curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

    #direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2
    
    if curvature > 2100 :#and abs(direction) < 100:
        road_inf = 'No Curve'
        curvature = -1
    elif curvature <= 2100 :#and direction < - 50:
        road_inf = 'Left Curve'
    elif curvature <= 2100:# and direction > 50:
        road_inf = 'Right Curve'
    else:
        if left_line.road_inf != None:
            road_inf = left_line.road_inf
            curvature = left_line.curvature
        else:
            road_inf = 'None'
            curvature = curvature

    center_lane = (right_line.startx + left_line.startx) / 2
    lane_width = right_line.startx - left_line.startx

    center_car = 720 / 2
    if center_lane > center_car:
        deviation =  str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
    elif center_lane < center_car:
        deviation =  str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
    else:
        deviation = str(0)
    left_line.road_inf = road_inf
    left_line.curvature = curvature
    left_line.deviation = deviation

    return road_inf, curvature, deviation


def print_road_status(img, left_line, right_line):
    """ print road status (curve direction, radius of curvature, deviation) """
    road_inf, curvature, deviation = road_info(left_line, right_line)
    cv2.putText(img, 'Road Status', (22, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (80, 80, 80), 2)

    lane_inf = 'Lane Info : ' + road_inf
    if curvature == -1:
        lane_curve = 'Curvature : Straight line'
    else:
        lane_curve = 'Curvature : {0:0.3f}m'.format(curvature)
    deviate = 'Deviation : ' + deviation

    cv2.putText(img, lane_inf, (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(img, lane_curve, (10, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
    cv2.putText(img, deviate, (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    return img

def print_road_map(image, left_line, right_line):      # right_line 삭제
    """ print simple road map """
    img = cv2.imread('D:/driving/driving/car_image2.jpg', -1)
    #random image
    img = cv2.resize(img, (120, 246))

    rows, cols = image.shape[:2]
    window_img = np.zeros_like(image)

    window_margin = left_line.window_margin
    left_plotx = left_line.allx
    right_plotx = right_line.allx
    ploty = left_line.ally
    lane_width = right_line.startx - left_line.startx
    lane_center = (right_line.startx + left_line.startx) / 2
    lane_offset = cols / 2 - (2*left_line.startx + lane_width) / 2
    car_offset = int(lane_center - 360)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_pts_l = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width - window_margin / 4, ploty]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - lane_width+ window_margin / 4, ploty])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array([np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty]))])
    right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset + window_margin / 4, ploty])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_pts]), (140, 0, 170))
    cv2.fillPoly(window_img, np.int_([right_pts]), (140, 0, 170))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([right_plotx + lane_offset - lane_width + window_margin / 4, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx + lane_offset - window_margin / 4, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

    #window_img[10:133,300:360] = img
    road_map = Image.new('RGBA', image.shape[:2], (0, 0, 0, 0))
    window_img = Image.fromarray(window_img)
    img = Image.fromarray(img)
    road_map.paste(window_img, (0, 0))
    #ValueError: bad transparency mask solve
    img = img.convert("RGBA")

    road_map.paste(img, (300-car_offset, 590), mask=img)
    road_map = np.array(road_map)
    road_map = cv2.resize(road_map, (95, 95))
    road_map = cv2.cvtColor(road_map, cv2.COLOR_BGRA2BGR)
    return road_map