# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from matplotlib.patches import Ellipse


# Function Definitions


# put entries in increasing angle order
def order_based_angle(x,y):

    rho = np.zeros((x.size,1))
    alpha = np.zeros((x.size,1))
    
    for ii in np.arange(0,x.size):
        
        rho[ii] = np.sqrt(x[ii]**2 + y[ii]**2)
        alpha[ii] = np.arctan2(y[ii],x[ii])
        
    polar = np.column_stack((alpha,rho))
    polar = polar[np.argsort(polar[:,0])]
    alpha = polar[:,0]
    rho = polar[:,1]
    
    for ii in np.arange(0,x.size):
        
        x[ii] = rho[ii]*np.cos(alpha[ii])
        y[ii] = rho[ii]*np.sin(alpha[ii])
        
    return x, y
        

# extract line features from local lidar point cloud
def extract_features(l_point_cloud_pc, x_true, x_bar):
    
    # eliminate measurements at infinity
    r1,t1 = l_point_cloud_pc[:,0], l_point_cloud_pc[:,1]
    temp =np.where(np.isinf(r1))
    r = np.delete(r1,temp)
    t = np.delete(t1,temp)         
    
    # point cloud in cartesian coordinates in the local frame
    l_point_cloud_xy = pol2cart(r, t)
    x = l_point_cloud_xy[:,0];
    y = l_point_cloud_xy[:,1];
    
    x, y = order_based_angle(x,y);

    # Split-and-Merge
    ## Step 1
    thresh = 0.1;
    S1 = np.column_stack((x,y));
    L = [S1];
    ## Step 2
    ii = 0;
    while ii < len(L):
        x1 = L[ii][0,0];
        y1 = L[ii][0,1];
        x2 = L[ii][(len(L[ii])-1),0];
        y2 = L[ii][(len(L[ii])-1),1];
        if len(L[ii]) > 2:
            if x2 == x1:
                m = 10000
            else:
                m = (y2 - y1)/(x2 - x1);
        else:
            ii += 1;
            continue;
        a = -m;
        b = 1;
        c = m*x1 - y1;
        d = np.zeros((len(L[ii]),1));
        for jj in np.arange(0,d.size):
            xjj = L[ii][jj,0];
            yjj = L[ii][jj,1];
            d[jj] = np.abs(a*xjj + b*yjj + c)/np.sqrt(a**2 + b**2);
        d_max = np.amax(d);
        jj_max = np.argmax(d);
        ## Step 4, 5
        if d_max <= thresh:
            ii += 1;
            continue
        else:
            Si1 = L[ii][0:jj_max + 1,:];
            Si2 = L[ii][jj_max:len(L[ii]),:];
            L[ii] = Si1;
            L.append(0);
            k = len(L) - 2;
            while k >= ii:
                L[k + 1] = L[k];
                k -= 1;
            L[ii + 1] = Si2;

    ## Step 8
    # Merge collinear segments
    ii = 0;
    while ii < (len(L) - 1): 
        x1 = L[ii][0,0];
        y1 = L[ii][0,1];
        x2 = L[ii + 1][(len(L[ii + 1])-1),0];
        y2 = L[ii + 1][(len(L[ii + 1])-1),1];
        if x2 == x1:
            m = 10000
        else:
            m = (y2 - y1)/(x2 - x1);
        a = -m;
        b = 1;
        c = m*x1 - y1;
        d = np.zeros((len(L[ii]),1));
        for jj in np.arange(0,d.size):
            xjj = L[ii][jj,0];
            yjj = L[ii][jj,1];
            d[jj] = np.abs(a*xjj + b*yjj + c)/np.sqrt(a**2 + b**2);
        d_max = np.amax(d);
        ## Step 4, 5
        if d_max > thresh:
            ii += 1;
            continue
        else:
            L[ii] = np.vstack((L[ii], L[ii + 1]));
            del L[ii + 1]
    
    # discard unreliable segments
    k = len(L) - 1;
    while k >= 0:
        xfirst = L[k][0,0];
        yfirst = L[k][0,1];
        xlast = L[k][len(L[k])-1,0];
        ylast = L[k][len(L[k])-1,1];
        length_seg = np.sqrt((ylast - yfirst)**2 + (xlast - xfirst)**2);
        if (len(L[k]) < 20) or (length_seg < 4*thresh) or (length_seg > 0.7 and len(L[k]) < 110):
            del L[k];
        k -= 1;
    
    # total Least-Sqauares for each segment to get the line parameters
    xL = np.zeros((len(L),1));
    yL = np.zeros((len(L),1));
    rho = np.zeros((len(L),1));
    alpha = np.zeros((len(L),1));
    xbar = np.zeros((len(L),1));
    ybar = np.zeros((len(L),1));
    
    # allocate memory to store intermediate line model parameters as in Eq. 5.18
    T = np.zeros((len(L),6))
    
    for ii in np.arange(0,len(L)):
        xh = L[ii][:,0];
        yh = L[ii][:,1];
        xbar[ii] = 1/len(L[ii])*np.sum(xh);
        ybar[ii] = 1/len(L[ii])*np.sum(yh);
        Sx2 = np.sum((xh - xbar[ii])**2);
        Sy2 = np.sum((yh - ybar[ii])**2);
        Sxy = np.sum((xh - xbar[ii])*(yh - ybar[ii]));
        alpha[ii] = 1/2*np.arctan2(-2*Sxy,(Sy2 - Sx2));
        rho[ii] = xbar[ii]*np.cos(alpha[ii]) + ybar[ii]*np.sin(alpha[ii]);
        if rho[ii] < 0:
            rho[ii] = -rho[ii];
            alpha[ii] = alpha[ii] + np.pi;
        xL[ii] = rho[ii]*np.cos(alpha[ii]);
        yL[ii] = rho[ii]*np.sin(alpha[ii]);
        
        # store the intermediate line model parameters for each line
        T[ii,0] = xbar[ii]
        T[ii,1] = ybar[ii]
        T[ii,2] = Sx2
        T[ii,3] = Sy2
        T[ii,4] = Sxy
        T[ii,5] = len(L[ii]) # n_i

    # plot the finite-length line segments corresponding to the point clusters 
    c = np.zeros((2,len(L)));
    for ii in np.arange(0,len(L)):
        c[:,ii] = np.hstack((xbar[ii],ybar[ii]))
        zp = np.zeros((2,len(L[ii])));
        for jj in np.arange(0,len(L[ii])):
            zp[:,jj] = np.hstack((L[ii][jj,0], L[ii][jj,1])) - c[:,ii];
        R = np.array([(np.cos(alpha[ii,0]), -np.sin(alpha[ii,0])), (np.sin(alpha[ii,0]), np.cos(alpha[ii,0]))]);
        zpp = np.transpose(R).dot(zp);
        e1pp = np.hstack((0, np.amax(zpp[1,:])));
        e2pp = np.hstack((0, np.amin(zpp[1,:])));
        e1 = R.dot(e1pp) + c[:,ii];
        e2 = R.dot(e2pp) + c[:,ii];
        e1_rotated = rotate(e1, (0,0), angle=x_true[2])
        e2_rotated = rotate(e2, (0,0), angle=x_true[2])
        plt.plot(np.hstack((e1_rotated[0], e2_rotated[0])) + x_true[0], np.hstack((e1_rotated[1], e2_rotated[1])) + x_true[1], color='k', lw=6, zorder=4);
    
    # allocate memory for the local feature map
    l_feature_m_pc = np.zeros((len(L),2))
    
    # save local feature map in polar coordinates (rho, alpha)
    for i in range(len(L)):
        
        l_feature_m_pc[i,0] = rho[i]
        l_feature_m_pc[i,1] = alpha[i] 
    
    l_feature_m_pc = fix_angle(l_feature_m_pc)
    
    plt.plot(-100, -100, 'k', label='Features', zorder=4, lw = 6)
    
    print(len(L), "line features have been detected from LiDAR scans.")
    print()
                   
    return l_feature_m_pc, T, L


# put angles between -pi to pi radian range
def fix_angle(a):
   
    for i in range(len(a)):
       
        if a[i,1] > np.pi:
           
            while True:
               
                a[i,1] = a[i,1] - np.pi*2
               
                if a[i,1] <= np.pi:
                   
                    break
       
        elif a[i,1] <= -np.pi:
           
            while True:
               
                a[i,1] = a[i,1] + np.pi*2
               
                if a[i,1] > -np.pi:
                   
                    break            
           
    return a  


# put angles between -pi to pi radian range
def fix_angle_one(a):
       
    if a > np.pi:
           
        while True:
               
            a = a - np.pi*2
               
            if a <= np.pi:
                   
                break
       
    elif a <= -np.pi:
           
        while True:
               
            a = a + np.pi*2
               
            if a > -np.pi:
                   
                break            
           
    return a 


# rotate p around the 'origin' 'angle' radians
# reference: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(p, origin=(0, 0), angle=0):
    
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


# initializes the map for the localization algorithm 
def set_map():
    
    # load a set of raw lidar point measurements in polar coordinates that represent the environment in the global frame
    g_point_cloud_pc = np.loadtxt("rectangular environment point cloud pc.out", delimiter=',')
    
    # load midpoints
    g_midpoints_xy = np.loadtxt("midpoints.out", delimiter=',')
    
    # put the measurements in -pi to pi radian range
    g_point_cloud_pc = fix_angle(g_point_cloud_pc)

    # load global frame line parameters (r, psi) 
    r = np.loadtxt("rho.out")
    psi = np.loadtxt("alpha.out")
    
    # construct global line landmark map in polar coordinates
    g_m_pc = np.array([r,psi]).T 
    # put the measurements in -pi to pi radian range
    g_m_pc = fix_angle(g_m_pc)

    # get global map in cartesian coordinates 
    g_m_xy = pol2cart(r, psi)
    
    return g_point_cloud_pc, g_m_pc, g_m_xy, g_midpoints_xy
    

# define control input    
def set_u(v,w):
    
    u = np.array([v,w])
    
    return u


# initializes the states
def set_x(x,y,theta):
    
    x = np.array([x,y,theta])
    
    return x


# process function (motion model) for non-zero angular velocity: Eq. 5.3
def motion_model(x, u):
     
    v = u[0]
    w = u[1]
    theta = x[2]
    
    A = np.array([(-v/w) * math.sin(theta) + (v/w) * math.sin(theta + w * DT), 
                  (v/w) * math.cos(theta) - (v/w) * math.cos(theta + w * DT),
                   w * DT]).T
    
    x_bar = x + A
    
    # put theta in -pi to pi range 
    x_bar[2] = fix_angle_one(x_bar[2])
    
    return x_bar


# jacobian of the process function (motion model) with respect to the state vector: Eq. 6.4
def jacobian_G(x, u):
     
    v = u[0]
    w = u[1]
    theta = x[2]
    
    j_G = np.array([[1, 0, (v/w) * (-math.cos(theta) + math.cos(theta + w * DT))],
                   [0, 1, (v/w) * (-math.sin(theta) + math.sin(theta + w * DT))],
                   [0, 0, 1]])
    
    return j_G


# jacobian of the process function (motion model) with respect to epsilon: Eq. 6.7
def jacobian_L(x, u):
    
    v = u[0]
    w = u[1]
    theta = x[2] 
    
    j_L = np.array([[(-math.sin(theta) + math.sin(theta + w * DT)) / w, v * (math.sin(theta) - math.sin(theta + w * DT)) / w**2 + v * DT * math.cos(theta + w * DT) / w, 0],
                     [(math.cos(theta) - math.cos(theta + w * DT)) / w, -v * (math.cos(theta) - math.cos(theta + w * DT)) / w**2 + v * DT * math.sin(theta + w * DT) / w, 0],
                     [0, DT, DT]])
    
    return j_L


# jacobian of the measurement model with respect to the state vector for all landmarks: Eq. 6.9
def jacobian_H(psi, k):
    
    # empty list to store j_H's
    j_H = []
    
    # for all landmarks
    for i in range(len(psi)):
        
        j_H_ = np.array([[math.pow(-1,k[i]) * math.cos(psi[i]), math.pow(-1,k[i]) * math.sin(psi[i]), 0],
                          [0, 0, -1]])
        
        j_H.append(j_H_)
    
    return j_H


# jacobian of the measurement model with respect to the state vector for all landmarks (midpoint): Eq. 6.17
def jacobian_H_mid(x_bar, g_midpoints_xy):
    
    x_ = x_bar[0] 
    y_ = x_bar[1] 
    
    # empty list to store j_H's
    H = []
    
    # for all landmarks
    for i in range(len(g_midpoints_xy)):
        
        H_ = np.array([[(x_ - g_midpoints_xy[i,0]) / (math.sqrt(math.pow(g_midpoints_xy[i,0] - x_,2) + math.pow(g_midpoints_xy[i,1] - y_,2))), (y_ - g_midpoints_xy[i,1]) / (math.sqrt(math.pow(g_midpoints_xy[i,0] - x_,2) + math.pow(g_midpoints_xy[i,1] - y_,2))), 0],
                        [(g_midpoints_xy[i,1] - y_) / (math.sqrt(math.pow(g_midpoints_xy[i,0] - x_,2) + math.pow(g_midpoints_xy[i,1] - y_,2))), (x_ - g_midpoints_xy[i,0]) / (math.sqrt(math.pow(g_midpoints_xy[i,0] - x_,2) + math.pow(g_midpoints_xy[i,1] - y_,2))), -1]])
        
        H.append(H_)
    
    return H


# predicted Sigma: Eq. 6.3
def predict_sigma(x, u, sigma, Q):
    
    j_G = jacobian_G(x,u)
    j_L = jacobian_L(x,u)
    
    sigma_bar = np.dot(j_G, sigma).dot(j_G.T) + np.dot(j_L,Q).dot(j_L.T)
    
    return sigma_bar
    

# Kalman filter prediction step: Eq. 4.19 and Eq. 4.20
def KF_prediction(x, u, sigma, Q):
        
    x_bar = motion_model(x, u)
        
    sigma_bar = predict_sigma(x, u, sigma, Q)

    return x_bar, sigma_bar


# Kalman filter prediction step: Eq. 4.19 and Eq. 4.20
def true_position(x_true, u):
    
    # noise to be added 
    noise1 = np.random.normal(0,0.05)/4
    noise2 = np.random.normal(0,0.04)/4 
    noise3 = np.random.normal(0,0.02)/4 
    
    x_true = motion_model(x_true, np.append(u, 0) + np.array([noise1,noise2, noise3]).T)
    
    return x_true


# calculate expected measurements (rho,alpha) of the landmarks in the body frame: Eq. 5.11
def measurement_model(x_bar, g_m_pc):
    
    number_of_landmarks = len(g_m_pc)
    
    # allocating memory
    z_hat = np.zeros((number_of_landmarks, 2)) 
    k = np.zeros(number_of_landmarks)
    
    x_ = x_bar[0] 
    y_ = x_bar[1] 
    theta = x_bar[2] 
    
    # global landmark parameters
    r = g_m_pc[:,0]
    psi = g_m_pc[:,1]

    #for all landmarks
    for i in range(number_of_landmarks):
       
        z_hat[i,0] = -r[i] + x_ * math.cos(psi[i]) + y_ * math.sin(psi[i])
        z_hat[i,1] = psi[i] - theta + math.pi
        if z_hat[i,0] >= 0:
            k[i] = 0
        else:
            z_hat[i,0] = -z_hat[i,0]
            z_hat[i,1] = z_hat[i,1] + math.pi
            k[i] = 1
       
        z_hat[i,1] = fix_angle_one(z_hat[i,1])
        
    return z_hat, k


# calculate expected measurements (rho,alpha) of the landmark midpoints in the body frame: Eq. 5.7
def midpoint_measurement_model(x, g_midpoints_xy):
    
    x_ = x[0] 
    y_ = x[1] 
    theta = x[2] 
    
    z_mid_hat = np.zeros((len(g_midpoints_xy), 2))
    
    for i in range(len(g_midpoints_xy)):
        
        z_mid_hat[i,0] = math.sqrt(math.pow(g_midpoints_xy[i,0] - x_, 2) + math.pow(g_midpoints_xy[i,1] - y_, 2)) 
        z_mid_hat[i,1] = math.atan2(g_midpoints_xy[i,1] - y_, g_midpoints_xy[i,0] - x_) - theta 
    
    # put the angle in -pi to pi range      
    z_mid_hat = fix_angle(z_mid_hat)
    
    return z_mid_hat


# convert from cartesian coordinates to polar coordinates
def cart2pol(x, y):
    
    d = np.sqrt(np.power(x,2) + np.power(y,2))
    betha = np.arctan2(y, x)
    
    a = np.array([d,betha])
    
    return a.T


# convert from polar coordinates to cartesian coordinates
def pol2cart(d, betha):
    
    x = d * np.cos(betha)
    y = d * np.sin(betha)
    
    a = np.array([x,y])
    
    return a.T


# a k value for each landmark is needed for picking the proper measurement model: Eq. 5.13
def get_k_1(x_bar, g_m_xy):
    
    x_ = x_bar[0] 
    y_ = x_bar[1] 
    theta = x_bar[2] 
    
    psi = np.arctan2(g_m_xy[:,1],g_m_xy[:,0])
    
    # allocating memory for k
    k = np.zeros(len(g_m_xy))
    
    # the origin
    origo = np.array([0,0])
    
    # slope of robot position vector
    if x_ != 0:
        m = y_ / x_
    else:
        m = 10000
        
    # allocating memory for the 2 unkown linear system coefficients in Eq. 5.13 (lower left and right corners)
    l = np.zeros((len(g_m_xy), 2)) 
    
    # for each landmark in the map
    for i in range(len(g_m_xy)):
        
        if psi[i] != 0:
            l[i,0] = -1/math.tan(psi[i])
        else:
            l[i,0] = 10000
        l[i,1] = -g_m_xy[i,1] - g_m_xy[i,0]/math.tan(psi[i])
        
        # if two lines are parallel to each other do not try to solve the linear system equation because it will throw an exception. Instead, set k equal to 1
        if m == l[i,0]:   
            k_ = 1
            
        else:
        
            # solving the linear system and finding the intersection (pink point on the righ side of Fig. 5.3)
            a = np.array([[m,-1], [l[i,0],-1]])
            b = np.array([0,l[i,1]])
            sol = np.linalg.solve(a,b)

            # if the intersection is part the position vector segment of the robot
            if np.sign(sol[0]) == np.sign(x_) and np.sign(sol[1]) == np.sign(y_) and np.linalg.norm(sol - origo) <= np.linalg.norm(x_bar[0:2] - origo):
                k_ = 0
            else:
                k_ = 1
        
        # save k
        k[i] = k_
        
    return k


# calculates a LiDAR scan (d, phi) for each point in the map with respect to the true state vector of the robot
def get_lidar_scan(g_point_cloud_pc, x_true, noise):
    
    x = x_true[0] 
    y = x_true[1] 
    
    # convert g_point_cloud_pc into cartesian coordinates in the global frame
    g_point_cloud_xy = pol2cart(g_point_cloud_pc[:,0], g_point_cloud_pc[:,1])
    # rotate point cloud and the robot pose so that lidar measurement at 0 angle is the one right in front of the robot
    g_point_cloud_xy_rotated = rotate(g_point_cloud_xy, (0,0), angle=-x_true[2])
    xy_rotated = rotate(np.array([x,y]), (0,0), angle=-x_true[2])
    
    # number of points in the map of the environment
    number_of_points = len(g_point_cloud_pc)
    
    # pre-allocating memory
    l_point_cloud_xy = np.zeros((0,2)) 
    
    # do the calculations for each point in the map
    for i in range(number_of_points):
        delta_x = g_point_cloud_xy_rotated[i,0] - xy_rotated[0] 
        delta_y = g_point_cloud_xy_rotated[i,1] - xy_rotated[1] 
        
        if math.pow(delta_x,2) + math.pow(delta_y,2) < 5:
            
            l_point_cloud_xy = np.append(l_point_cloud_xy,[[delta_x, delta_y]], axis = 0)

    # convert the simulated LiDAR measurements into cartesian coordinates
    l_point_cloud_pc = cart2pol(l_point_cloud_xy[:,0], l_point_cloud_xy[:,1])
    
    l_point_cloud_pc = l_point_cloud_pc[np.argsort(l_point_cloud_pc[:,1])]
    
    j = len(l_point_cloud_pc) - 1
    while j >= 0:  

        if l_point_cloud_pc[j,0] - l_point_cloud_pc[j-1,0] > 0.03 and l_point_cloud_pc[j,1] - l_point_cloud_pc[j-1,1] < 0.04:

            l_point_cloud_pc = np.delete(l_point_cloud_pc, j, 0)
            if j >= len(l_point_cloud_pc):
                j = j - 1
            continue
                
        j = j - 1
    
    if noise == 1:
    
        for i in range(len(l_point_cloud_pc)):
        
            # pick a sample from a noise distribution with 0 mean and 0.01 standard deviation to be added into radius of the scans to mimic real world implementation
            noise = np.random.normal(0,0.01) 
            l_point_cloud_pc[i,0] =  l_point_cloud_pc[i,0] + noise
    
    return l_point_cloud_pc


# calculate midpoints of the extracted features
def get_midpoint(L):

    number_of_features = len(L)
    
    l_midpoints_xy = np.zeros([number_of_features,2])
    
    for i in range(number_of_features):
        
        number_of_points = len(L[i])
        
        for j in range(number_of_points):
                
            l_midpoints_xy[i,0] = l_midpoints_xy[i,0] + L[i][j][0] 
            l_midpoints_xy[i,1] = l_midpoints_xy[i,1] + L[i][j][1]
                
        l_midpoints_xy[i,0] = l_midpoints_xy[i,0] / number_of_points 
        l_midpoints_xy[i,1] = l_midpoints_xy[i,1] / number_of_points    
    
    return l_midpoints_xy


# section 5.3.3: Line Feature Covariance Estimation
def estimate_line_feature_covariance(T, L, alpha, midpoint_flag = 0):
    
    # empty list to store R_i's
    R = []
    
    # for each line feature
    for i in range(len(T)): 
        
        # empty lists to store A_i, B_i and D_i's
        R_ = np.zeros((2,2))
        
        # intermediate line feature parameters to be used in covariance estimation
        x_bar = T[i][0]
        y_bar = T[i][1]
        Sx_2 = T[i][2]
        Sy_2 = T[i][3]
        Sxy = T[i][4]
        n = T[i][5]
        alpha_ = alpha[i]
    
        # for each point in the line feature
        for j in range(len(L[i])): 
            
            # nomenclature --> L[0][1][0] = [line 0][point 1][x value]
        
            # get xy coordinate of point j of line i
            x_h = L[i][j][0]
            y_h = L[i][j][1]
            
            # get polar coordinate of point j of line i (to be used in Eq. 5.27)
            p_pc = cart2pol(x_h, y_h)
            d = p_pc[0]
            phi = p_pc[1]        
            
            if midpoint_flag == 0:
            
                # Eq. 5.24
                A21 = ((y_bar - y_h) * (Sy_2 - Sx_2) + (2 * Sxy * (x_bar - x_h))) / (math.pow(Sy_2 - Sx_2,2) + 4 * math.pow(Sxy,2))
                A11 = (math.cos(alpha_) / n) - A21 * (x_bar * math.sin(alpha_) - y_bar * math.cos(alpha_))
                A22 = ((x_bar - x_h) * (Sy_2 - Sx_2) + (2 * Sxy * (y_bar - y_h))) / (math.pow(Sy_2 - Sx_2,2) + 4 * math.pow(Sxy,2))
                A12 = (math.sin(alpha_) / n) - A22 * (x_bar * math.sin(alpha_) - y_bar * math.cos(alpha_))
            
            else:
                
                # Eq. 6.20
                A11 = x_bar / (n * math.sqrt(math.pow(y_bar,2) + math.pow(x_bar,2)))
                A12 = y_bar / (n * math.sqrt(math.pow(y_bar,2) + math.pow(x_bar,2)))
                A21 = -y_bar / (n * (math.pow(y_bar,2) + math.pow(x_bar,2)))
                A22 = -x_bar / (n * (math.pow(y_bar,2) + math.pow(x_bar,2)))
            
            # Eq. 5.23
            A_ = np.array([[A11,A12],
                            [A21,A22]])
            
            # Eq. 5.27
            B_ = np.array([[math.cos(phi), -d * math.sin(phi)],
                            [math.sin(phi), d * math.cos(phi)]])
            
            # Eq. 5.25
            # D should be calculated as a function of distance and angle corresponding to a point measurement
            D_ = np.array([[0.01**2,0], 
                            [0,0]])
        
            # Eq. 5.28
            R_ = R_ + np.dot(A_, B_).dot(D_).dot(B_.T).dot(A_.T)
            
        R.append(R_)
    
    
    return R


# S is needed to compute the Kalman gain: Eq. 6.10
def get_innovation_covariance(j_H, sigma_bar, R, i, number_of_landmarks):
    
    # allocating memory
    S = []
    
    # for all landmarks
    for j in range(number_of_landmarks):
        
        # R[i] is the line feature covariance matrix for the i'th feature
        S_ = np.dot(j_H[j], sigma_bar).dot(j_H[j].T) + R[i]
        
        S.append(S_)
    
    return S


# calculates Mahalanobis distance and nearest neighbour value to determine the correspondence of the feature
def get_correspondence(z_i, z_hat, S, assigned_landmarks, z_i_mid, z_mid_hat, S_mid):
    
    flag = 0
    correspondence = -1
    number_of_landmarks = len(z_hat)
    
    # allocating memory
    Mahalanobis_distance = np.empty(number_of_landmarks)
    nearest_neighbour = np.empty(number_of_landmarks)
    
    # allocating memory
    Mahalanobis_distance_mid = np.empty(number_of_landmarks)
    nearest_neighbour_mid = np.empty(number_of_landmarks)
    
    # variable to store minimum nearest neighbour value
    minimum = np.inf
    
    # validation gate threshold
    threshold = 6
    
    # for all landmarks
    for i in range(number_of_landmarks):
        
        # values needed in the calculations below
        delta = z_i - z_hat[i]
        inverse_S = np.linalg.inv(S[i])
        
        # calculate Mahalobis distances from line feature z_i to all landmarks
        # Eq. 6.14
        Mahalanobis_distance[i] = np.dot(delta.T,inverse_S).dot(delta) 
        
        # applying validation gate (rejecting possible outlier correspondence)
        if Mahalanobis_distance[i] > threshold:
            
            print("Landmark ", i, ": Too large Mahalanobis distance: ", Mahalanobis_distance[i])
            continue
 
        # calculate nearest neighbour values from line feature z_i to all landmarks
        # Eq. 6.13
        nearest_neighbour[i] = Mahalanobis_distance[i] + np.log(np.linalg.det(S[i]))
        
        # values needed in the calculations below (midpoint)
        delta_mid = z_i_mid - z_mid_hat[i]
        inverse_S_mid = np.linalg.inv(S_mid[i])
        
        # calculate Mahalobis distances from line feature z_i to all landmarks (midpoint)
        # Eq. 6.21 (first part)
        Mahalanobis_distance_mid[i] = np.dot(delta_mid.T,inverse_S_mid).dot(delta_mid) 

        # calculate nearest neighbour values from line feature z_i to all landmarks
        # Eq. 6.21
        nearest_neighbour_mid[i] = Mahalanobis_distance_mid[i] + np.log(np.linalg.det(S_mid[i]))
                
        # finding the correspondence that gives the minimum cost value using the weighted sum
        cost =  0.1 * nearest_neighbour[i] + 2 * nearest_neighbour_mid[i]
        
        # finding the correspondence that gives the minimum nearest_neighbour value
        if cost < minimum:
                
            for k in range(len(assigned_landmarks)):

                if i == assigned_landmarks[k]:

                    flag = 1
                    break

            if flag == 0:

                minimum = cost
                correspondence = i
                
            flag = 0
        
        print("Landmark ", i, "-> ", "M: ", Mahalanobis_distance[i], " NN: ", nearest_neighbour[i], " C: ", correspondence, " NN_mid: ", nearest_neighbour_mid[i])
    
    print("Final: ", "NN: ", minimum, " C: ", correspondence)
    
    return correspondence


# compute Kalman gain as shown in line 19 of algorithm 2  
def get_Kalman_gain(sigma_bar,j_H,S):
    
    Kalman_gain = np.dot(sigma_bar,j_H.T).dot(np.linalg.inv(S))           
    
    return Kalman_gain

                
# plots error ellipse of a 2D Gaussian                
def plot_position_error_ellipse(x, sigma):
   
    sigma_ = sigma[0:2,0:2]
    lambda_, v = np.linalg.eig(sigma_)
    lambda_ = np.sqrt(lambda_)
    x_ellipse = Ellipse((x[0], x[1]), width=lambda_[0]*5*2, height=lambda_[1]*5*2, angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), zorder = 9, ec = 'k', fc = 'gold', lw = 2)
    plt.gca().add_artist(x_ellipse)


    
############# Localization Algorithm #############

    
# Define constants

pose_shape = np.array([[-0.1,0.1], [-0.1,-0.1], [0.2,0]])
scale = 1
DT = 1 / scale # delta t [s] (time step) 

# angular velocity commands
w = [1, 2, 0.3, 3, -1, 1.5, -1.4, -1, 0.1, -1.3, 2, 2, 1.5, 0.2, -0.5, -1, -0.1, 1, 2.5, -2, 0.5, 2, 3.55, -0.95, -0.95, -0.65, -0.05, -2.5, 1.85, -0.4, -0.9, -0.1, -0.3, 1.45, -0.1, 0.9, 1, 0.4, 1, -3.25, 0.2, 0.7, 0.4, 2]
w = [x/4 for x in w]

# state covariance 
sigma = np.diag([0.1, 0.1, 0])**2 

# process covariance
alpha_1 = 0.1
alpha_2 = 0.1
alpha_3 = 0.1
alpha_4 = 0.1
alpha_5 = 0.1
alpha_6 = 0.1
#Q = np.diag([alpha_1 * v**2 + alpha_2 * w**2, alpha_3 * v**2 + alpha_4 * w**2, alpha_5 * v**2 + alpha_6 * w**2])**2 
Q = (np.diag([0.05, 0.04, 0.02])/4)**2 

# define state vector x
x = set_x(1.5,2,-np.pi/2) # (x,y,theta)

# true state vector (for performance evaluation)
x_true = set_x(1.5,2,-np.pi/2)

# odometry vector
x_odom = set_x(1.5,2,-np.pi/2)

# load the map of the environment: global point cloud in polar coordinates and polar/cartesian coordinates of the line landmarks in global frame
g_point_cloud_pc, g_m_pc, g_m_xy, g_midpoints_xy = set_map()

# number of landmarks in the map
number_of_landmarks = len(g_m_pc)

# figure number of the animation
figure = 1

# global point cloud of the environment in cartesian coordinates
g_point_cloud_xy = pol2cart(g_point_cloud_pc[:,0], g_point_cloud_pc[:,1])

# number of iterations to run the algorithm
number_of_iterations = int(5 * len(w) * scale)

# lists with zero entries to store robot's estimated and ground truth trajectories
trajectory_estimated = np.zeros((number_of_iterations+1, 3))
trajectory_true = np.zeros((number_of_iterations+1, 2))
trajectory_odom = np.zeros((number_of_iterations+1, 2))

# initial position belief
trajectory_estimated[0,0] = x[0]
trajectory_estimated[0,1] = x[1]
trajectory_estimated[0,2] = x[2]

# initial true position
trajectory_true[0,0] = x_true[0]
trajectory_true[0,1] = x_true[1]

# initial motion position
trajectory_odom[0,0] = x_odom[0]
trajectory_odom[0,1] = x_odom[1]

# variables for counting
u_id = 0
tr = 0

# start the algorithm
for j in range(number_of_iterations):
    
    if j % (5 * scale) == 0:
        # define control input vector u
        u = set_u(1/4,w[u_id]) 
        u_id = u_id + 1
    
    print("Iteration:", j+1)
    
    # figure settings for the animation
    fig = plt.figure(figsize=(22,13.6))
    ax = plt.gca()
    plt.grid(color='black', linestyle='-', lw = 0.25, alpha = 0.2, zorder=1)
    gain=1.30
    ax.set_xlim(-5.5*gain, 5.5*gain)
    ax.set_ylim(-3.75*gain, 3.75*gain)
    plt.xticks([-6,-4,-2,0,2,4,6],[-6,-4,-2,0,2,4,6],fontsize=35)
    plt.yticks([-4,-2,0,2,4],fontsize=35)
    plt.scatter(1.5,2,zorder=8,color='m',s=80)
    
    # plot the global point cloud of the environment
    plt.scatter(g_point_cloud_xy[:,0],g_point_cloud_xy[:,1], s=15, color='dodgerblue', zorder=2)
    
    # get odometry data
    x_odom = motion_model(x_odom,u)
    trajectory_odom[j+1,0] = x_odom[0]
    trajectory_odom[j+1,1] = x_odom[1]
    
    # calculate x belief using the motion model 
    x_bar, sigma_bar = KF_prediction(x, u, sigma, Q)
    
    # calculate x ground truth
    x_true = true_position(x_true, u)
    # saving it for the animation
    trajectory_true[j+1,0] = x_true[0]
    trajectory_true[j+1,1] = x_true[1]

    pose_shape_rotated = rotate(pose_shape, (0,0), angle=x_true[2])
    t1 = plt.Polygon(pose_shape_rotated + x_true[0:2], color='lime', zorder=6)
    plt.gca().add_patch(t1)
    
    pose_shape_rotated = rotate(pose_shape, (0,0), angle=x_odom[2])
    t3 = plt.Polygon(pose_shape_rotated + x_odom[0:2], color='C1', zorder=5)
    plt.gca().add_patch(t3)
    
    if j > 29:
        tr = j - 29

    # get LiDAR measurements from the body frame in polar and cartesian coordinates
    l_point_cloud_pc = get_lidar_scan(g_point_cloud_pc, x_true,1)
    l_point_cloud_xy = pol2cart(l_point_cloud_pc[:,0],l_point_cloud_pc[:,1])
    
    if len(l_point_cloud_pc) == 0:
        
        x = x_bar
        sigma = sigma_bar

        # plot error ellipse robot position
        plot_position_error_ellipse(x, sigma)  

        # saving estimated position for the animation
        trajectory_estimated[j+1,0] = x[0]
        trajectory_estimated[j+1,1] = x[1]
        trajectory_estimated[j+1,2] = x[2]
        
        pose_shape_rotated = rotate(pose_shape, (0,0), angle=x_bar[2])
        t2 = plt.Polygon(pose_shape_rotated + x_bar[0:2], color='m', zorder=7)
        plt.gca().add_patch(t2)

        # plotting the true and estimated trajectories
        plt.plot(trajectory_estimated[tr:j+2,0],trajectory_estimated[tr:j+2,1],'m', linestyle = '--', zorder=7, label='EKF', lw = 6)
        plt.plot(trajectory_true[tr:j+2,0],trajectory_true[tr:j+2,1],'lime', zorder=6, label='True', lw = 6)
        plt.plot(trajectory_odom[tr:j+2,0],trajectory_odom[tr:j+2,1],'C1', zorder=5, label='Motion model', lw = 6)
        
        plt.xlabel("$x_\mathcal{G}$ [m]", rotation=0, labelpad= 5, fontsize = 35)    
        plt.ylabel("$y_\mathcal{G}$ [m]", rotation=90, labelpad= 5, fontsize = 35) 
    
        plt.plot(-100, -100, 'k', label='Features', zorder=4, lw = 6)
        plt.plot(-100, -100, 'mediumseagreen', label='Correspondences', zorder=4, lw = 6)
        
        plt.legend(mode='expand', loc='upper right', ncol=6, fontsize=30)

        # saving figures for animation
        my_path = os.getcwd() + "/localization figures/"
        my_file = str(figure) + '.svg'
        plt.savefig(os.path.join(my_path, my_file))
        figure = figure + 1 
        plt.close()
        
        continue
    
    # uncomment below if you want to include the noisy lidar measurements in the animation figures
    l_point_cloud_xy_rotated = rotate(l_point_cloud_xy, (0,0), angle=x_true[2])
    plt.scatter(l_point_cloud_xy_rotated[:,0]+x_true[0],l_point_cloud_xy_rotated[:,1]+x_true[1], color='r',zorder=3, label='LiDAR', s=100)

    # get local map of the line features in polar coordinates (rho, alpha) that are extracted from LiDAR scans
    l_feature_m_pc, T, L = extract_features(l_point_cloud_pc, x_true, x_bar) 
    # T contains the intermediate line feature model parameters as in Eq. 5.20
    # L contains the cartesian coordinates of all points belonging to line features

    l_feature_m_pc_xy = pol2cart(l_feature_m_pc[:,0],l_feature_m_pc[:,1])
    
    # compute line feature covariances
    alpha = l_feature_m_pc[:,1]
    R = estimate_line_feature_covariance(T, L, alpha)
    
    # compute midpoint covariances
    R_mid = estimate_line_feature_covariance(T, L, alpha, 1)
    
    # empty list to hold the landmarks that are assigned to features
    assigned_landmarks = []

    number_of_features = len(R)
    
    assigned_features = []

    # for all features
    for i in range(number_of_features):

        # get expected landmark measurements
        z_hat, k = measurement_model(x_bar, g_m_pc)

        z_hat_xy = pol2cart(z_hat[:,0],z_hat[:,1])

        # get expected midpoint measurements
        z_mid_hat = midpoint_measurement_model(x_bar, g_midpoints_xy)

        # calculate jacobian of the measurement model
        psi = g_m_pc[:,1]
        j_H = jacobian_H(psi, k)

        # calculate jacobian of the midpoint measurement model
        j_H_mid = jacobian_H_mid(x_bar, g_midpoints_xy)

        # calculate innovation covariances between i'th line feature and all landmarks in the map
        S = get_innovation_covariance(j_H, sigma_bar, R, i, number_of_landmarks)
                  
        # get i'th line feature local frame parameters (rho,alpha)
        z_i = l_feature_m_pc[i]

        # calculate innovation covariances between i'th line feature midpoint and all landmark midpoints in the map
        S_mid = get_innovation_covariance(j_H_mid, sigma_bar, R_mid, i, number_of_landmarks)
        
        # get i'th line feature midpoint local frame parameters (rho_mid,alpha_mid)
        z_i_mid = cart2pol(T[:,0], T[:,1])

        print("feature: ",i)
        print("assigned_landmarks:")
        for m in range(len(assigned_landmarks)):
            print(assigned_landmarks[m])
            
        # find which landmark i'th feature corresponds to
        correspondence = get_correspondence(z_i, z_hat, S, assigned_landmarks, z_i_mid[i], z_mid_hat, S_mid)

        # if no correspondence has been found, don't compute the Kalman gain and don't update x_bar and sigma_bar 
        if correspondence == -1:
            continue
            
        assigned_features.append(i)
        
        # include the recent correspondence in the list
        assigned_landmarks.append(correspondence)
            
        # compute the Kalman gain
        Kalman_gain = get_Kalman_gain(sigma_bar,j_H[correspondence],S[correspondence])

        # calculate the innovation between the landmark and the corresponding feature
        innovation = z_i - z_hat[correspondence]
        # update x_bar and sigma_bar with the knowledge obtained from detecting a landmark
        # line 20 of algorithm 2
        x_bar = x_bar + np.dot(Kalman_gain, innovation)
        # line 21 of algorithm 2
        
        A = np.eye(3,3) - np.dot(Kalman_gain, j_H[correspondence])
        sigma_bar = np.dot(A, sigma_bar)   
        x_bar[2] = fix_angle_one(x_bar[2])

    print()
    
    # update variables before starting the a new iteration: lines 23 and 24 of algorithm 2    
    x = x_bar 
    sigma = sigma_bar
    
    # saving estimated position for the animation
    trajectory_estimated[j+1,0] = x[0]
    trajectory_estimated[j+1,1] = x[1]
    trajectory_estimated[j+1,2] = x[2]
    
    pose_shape_rotated = rotate(pose_shape, (0,0), angle=x_bar[2])
    t2 = plt.Polygon(pose_shape_rotated + x_bar[0:2], color='m',zorder=7)
    plt.gca().add_patch(t2)
        
    # plotting the true and estimated trajectories
    plt.plot(trajectory_estimated[tr:j+2,0],trajectory_estimated[tr:j+2,1],'m', linestyle = '--', zorder=7, label='EKF', lw=6)
    plt.plot(trajectory_true[tr:j+2,0],trajectory_true[tr:j+2,1],'lime', zorder=6, label='True', lw=6)
    plt.plot(trajectory_odom[tr:j+2,0],trajectory_odom[tr:j+2,1],'C1', zorder=5, label='Motion model', lw=6)
    
    ptr = 0
    for i in assigned_landmarks:
        
        if i == 0 or i == 1 or i == 2 or i == 3:
            c = rotate([T[assigned_features[ptr],0], T[assigned_features[ptr],1]], (0,0), angle=x_true[2])
            plt.plot([x_bar[0], x_true[0] + c[0]],[x_bar[1], x_true[1] + c[1]], color='mediumseagreen', lw=6) 
            ptr += 1
        else:
            plt.plot([x_bar[0], g_midpoints_xy[i,0]], [x_bar[1], g_midpoints_xy[i,1]], color='mediumseagreen', lw=6) 
            ptr += 1

    plt.plot(-100, -100, color='mediumseagreen', label='Correspondences', zorder=4, lw = 6)    
    
    # plot error ellipse robot position
    plot_position_error_ellipse(x, sigma) 
    
    plt.xlabel("$x_\mathcal{G}$ [m]", rotation=0, labelpad= 5, fontsize = 35)    
    plt.ylabel("$y_\mathcal{G}$ [m]", rotation=90, labelpad= 5, fontsize = 35) 
    fig.tight_layout(pad=0) 

    plt.legend(mode='expand', loc='upper right', ncol = 6, fontsize=30)
    # saving figures for animation
    my_path = os.getcwd() + "/localization figures midpoint/"
    my_file = str(figure) + '.png'
    plt.savefig(os.path.join(my_path, my_file))
    figure = figure + 1 
    plt.close()

