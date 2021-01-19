#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import sys
from matplotlib.patches import Ellipse

# In[2]:

''' Function Definitions '''

# Order LiDAR points in increasing angle order
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

# Extract line features from local lidar point cloud
def extract_features(l_point_cloud_pc, x_true, x_bar):
    
    # Eliminate measurements at infinity
    r1,t1 = l_point_cloud_pc[:,0], l_point_cloud_pc[:,1]
    temp = np.where(np.isinf(r1))
    r = np.delete(r1,temp)
    t = np.delete(t1,temp)         
    
    # Point cloud in cartesian coordinates in the local frame
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
                m = 10000;
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
    #L.append(L[0]);
    ii = 0;
    while ii < (len(L) - 1): 
        x1 = L[ii][0,0];
        y1 = L[ii][0,1];
        x2 = L[ii + 1][(len(L[ii + 1])-1),0];
        y2 = L[ii + 1][(len(L[ii + 1])-1),1];
        if len(L[ii]) > 2:
            if x2 == x1:
                m = 10000;
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
            del L[ii + 1];
            #if ii == (len(L)-1):
               #del L[0];
    
    # Discard sgements that are too short and unreliable
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

    # Total Least-Sqauares for each segment to get the line parameters
    xL = np.zeros((len(L),1));
    yL = np.zeros((len(L),1));
    rho = np.zeros((len(L),1));
    alpha = np.zeros((len(L),1));
    xbar = np.zeros((len(L),1));
    ybar = np.zeros((len(L),1));
    
    # Allocate memory to store intermediate line model parameters
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
        
        # Store the intermediate line model parameters for each line
        T[ii,0] = xbar[ii]
        T[ii,1] = ybar[ii]
        T[ii,2] = Sx2
        T[ii,3] = Sy2
        T[ii,4] = Sxy
        T[ii,5] = len(L[ii])

    # Plot the finite-length line segments corresponding to the point clusters 
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
        e1_rotated = rotate(e1, (0,0), angle = x_true[2])
        e2_rotated = rotate(e2, (0,0), angle = x_true[2]) 
        plt.plot(np.array([e1_rotated[0], e2_rotated[0]]) + x_true[0], np.array([e1_rotated[1], e2_rotated[1]]) + x_true[1], color = 'k', lw = 6, zorder = 4);
    
    plt.plot(-100, -100, color = 'k', lw = 6, zorder = 4, label = 'Features')
    
    # Allocate memory for the local feature map
    l_feature_m_pc = np.zeros((len(L),2))
    
    # Save local feature map in polar coordinates (rho, alpha)
    for i in range(len(L)):
        
        l_feature_m_pc[i,0] = rho[i]
        l_feature_m_pc[i,1] = alpha[i] 
    
    # Keep angle measuremnet between -pi and pi
    l_feature_m_pc = fix_angle(l_feature_m_pc)   
    
    print(len(L), "line features have been detected from LiDAR scans.")
    print()
                   
    return l_feature_m_pc, T, L


# Keep a vector of angles between -pi and pi
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


# Keep a single angle between -pi and pi
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


# Rotate p around the 'origin' 'angle' radians
# reference: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(p, origin=(0, 0), angle=0):
    
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


# Initializes the map for the localization algorithm 
def set_map():
    
    # Load a set of raw LiDAR point measurements in polar coordinates that represent the environment in the global frame
    g_point_cloud_pc = np.loadtxt("rectangular environment point cloud pc.out", delimiter=',')  # add path here
    g_point_cloud_pc = fix_angle(g_point_cloud_pc)

    # Load global frame parameters (r, psi) and midpoint cooridnates in the environment
    r = np.loadtxt("C:/Users/Matei CACIULEANU/Desktop/University/Semester 6/P6/Python/SLAM/rho.out")
    psi = np.loadtxt("C:/Users/Matei CACIULEANU/Desktop/University/Semester 6/P6/Python/SLAM/alpha.out")
    g_m_mid_true = np.loadtxt("C:/Users/Matei CACIULEANU/Desktop/University/Semester 6/P6/Python/SLAM/midpoints.out", delimiter=',')
    
    # Construct global line landmark map in polar coordinates
    g_m_pc_true = np.array([r,psi]).T 
    g_m_pc_true = fix_angle(g_m_pc_true)

    # Get global map in cartesian coordinates 
    g_m_xy_true = pol2cart(r, psi)
    
    return g_point_cloud_pc, g_m_pc_true, g_m_xy_true, g_m_mid_true
    

# Define control input    
def set_u(v,w):
    
    u = np.array([v,w])
    
    return u


# Initializes the pose
def set_x(x,y,theta):
    
    x = np.array([x,y,theta])
    
    return x


# Process function (motion model)
def motion_model(x, u):
     
    v = u[0]
    w = u[1]
    theta = x[2]
    
    A = np.array([(-v/w) * math.sin(theta) + (v/w) * math.sin(theta + w * DT), 
                  (v/w) * math.cos(theta) - (v/w) * math.cos(theta + w * DT),
                   w * DT]).T
    
    x_bar = x + A
    
    x_bar[2] = fix_angle_one(x_bar[2])
    
    return x_bar


# Jacobian of the process function with respect to the pose vector
def jacobian_G(x, u):
     
    v = u[0]
    w = u[1]
    theta = x[2]
    
    j_G = np.array([[1, 0, (v/w) * (-math.cos(theta) + math.cos(theta + w * DT))],
                    [0, 1, (v/w) * (-math.sin(theta) + math.sin(theta + w * DT))],
                    [0, 0, 1]])
    
    return j_G


# Jacobian of the process function with respect to epsilon
def jacobian_L(x, u):
    
    v = u[0]
    w = u[1]
    theta = x[2] 
    
    j_L = np.array([[(-math.sin(theta) + math.sin(theta + w * DT)) / w, v * (math.sin(theta) - math.sin(theta + w * DT)) / w**2 + v * DT * math.cos(theta + w * DT) / w, 0],
                    [(math.cos(theta) - math.cos(theta + w * DT)) / w, -v * (math.cos(theta) - math.cos(theta + w * DT)) / w**2 + v * DT * math.sin(theta + w * DT) / w, 0],
                    [0, DT, DT]])
    
    return j_L


# Jacobian of the measurement model with respect to the pose and individual landmarks
def jacobian_H(x_bar, psi, k):
   
    x = x_bar[0] 
    y = x_bar[1]
    
    j_H = [] 
    j_H_ = np.zeros((5,2))
    
    # for all landmarks
    for i in range(len(psi)):
        
        j_Hx_ = np.array([[math.pow(-1,k[i]) * math.cos(psi[i]), math.pow(-1,k[i]) * math.sin(psi[i]), 0],
                          [0, 0, -1]])
        j_Hm_ = np.array([[math.pow(-1,k[i] + 1), math.pow(-1,k[i]) * (-x*math.sin(psi[i]) + y * math.cos(psi[i]))],
                          [0, 1]])
        
        j_H_ = np.concatenate((j_Hx_, j_Hm_), axis = 1)
        j_H.append(j_H_)
    
    return j_H


# Kalman covaraiance prediction
def predict_sigma(x, u, sigmaxx, sigmaxm, Q):
    
    j_G = jacobian_G(x,u)
    j_L = jacobian_L(x,u)
    
    sigmaxx_bar = np.dot(j_G, sigmaxx).dot(j_G.T) + np.dot(j_L,Q).dot(j_L.T)
    sigmaxm_bar = np.dot(j_G, sigmaxm)
    
    return sigmaxx_bar, sigmaxm_bar


# Kalman prediction step
def KF_prediction(x, u, sigmaxx, sigmaxm, Q):
        
    x_bar = motion_model(x, u)
    
    sigmaxx_bar, sigmaxm_bar = predict_sigma(x, u, sigmaxx, sigmaxm, Q)

    return x_bar, sigmaxx_bar, sigmaxm_bar


# True, noisy robot motion
def true_position(x_true, u):
    
    # Noise to be added (specifiy mean standard deviation)
    noise1 = np.random.normal(0,0.05)/4
    noise2 = np.random.normal(0,0.04)/4 
    noise3 = np.random.normal(0,0.02)/4 
    
    x_true = motion_model(x_true, np.append(u, 0) + np.array([noise1, noise2, noise3]).T)

    return x_true


# Calculate expected measurements (rho,alpha) of the landmarks in the body frame
def measurement_model(x_bar, g_m_pc):
       
    number_of_landmarks = len(g_m_pc)
    k = np.zeros(number_of_landmarks)
    z_hat = np.zeros((number_of_landmarks, 2)) 
    
    x_ = x_bar[0] 
    y_ = x_bar[1] 
    theta = x_bar[2] 
    
    r = g_m_pc[:,0]
    psi = g_m_pc[:,1]
    
    # for all landmarks
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


# Calculate expected global-frame polar coordinates (r_N+1, pis_N+1) of the new landmark
def inverse_measurement_model(x_bar, z):
    
    x = x_bar[0] 
    y = x_bar[1] 
    theta = x_bar[2] 
    
    psi = z[1] + theta - math.pi
    r = -z[0] + x * math.cos(psi) + y * math.sin(psi)
    if r >= 0:
        k = 0
    else:
        r = -r
        psi = psi + math.pi
        k = 1
        
    # put the angle in -pi to pi range      
    psi = fix_angle_one(psi)
    
    return r, psi, k


# Convert from Cartesian coordinates to polar coordinates
def cart2pol(x, y):
    
    d = np.sqrt(np.power(x,2) + np.power(y,2))
    betha = np.arctan2(y, x)
    
    a = np.array([d,betha])
    
    return a.T


# Convert from polar coordinates to Cartesian coordinates
def pol2cart(d, betha):
    
    x = d * np.cos(betha)
    y = d * np.sin(betha)
    
    a = np.array([x,y])
    
    return a.T


# Calculates a LiDAR scan (d, phi) for each point in the map with respect to the true state vector of the robot
def get_lidar_scan(g_point_cloud_pc, x_true, noise):
    
    x = x_true[0]
    y = x_true[1]
   
    # Convert g_point_cloud_pc into Cartesian coordinates in the global frame
    g_point_cloud_xy = pol2cart(g_point_cloud_pc[:,0], g_point_cloud_pc[:,1])
    # Rotate point cloud and the robot pose so that lidar measurement at 0 angle is the one right in front of the robot
    g_point_cloud_xy_rotated = rotate(g_point_cloud_xy, (0,0), angle=-x_true[2])
    xy_rotated = rotate(np.array([x,y]), (0,0), angle=-x_true[2])
   
    # Number of points in the map of the environment
    number_of_points = len(g_point_cloud_pc)
   
    l_point_cloud_xy = np.zeros((0,2))
   
    # Restrict the LiDAR range to a specified radius
    for i in range(number_of_points):
        delta_x = g_point_cloud_xy_rotated[i,0] - xy_rotated[0]
        delta_y = g_point_cloud_xy_rotated[i,1] - xy_rotated[1]
       
        if math.pow(delta_x,2) + math.pow(delta_y,2) < 5:
           
            l_point_cloud_xy = np.append(l_point_cloud_xy,[[delta_x, delta_y]], axis = 0)

    # Convert the simulated LiDAR measurements into cartesian coordinates and sort them by the angle
    l_point_cloud_pc = cart2pol(l_point_cloud_xy[:,0], l_point_cloud_xy[:,1])
    l_point_cloud_pc = l_point_cloud_pc[np.argsort(l_point_cloud_pc[:,1])]
    
    # Avoid seeing through walls and add measurmeent noise
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
       
            # Pick a sample from a noise distribution to be added into radius of the scans to mimic real-world implementation
            noise = np.random.normal(0,0.01)
            l_point_cloud_pc[i,0] =  l_point_cloud_pc[i,0] + noise
    
    return l_point_cloud_pc


# Line feature covariance estimation
def estimate_line_feature_covariance(T, L, alpha):
    
    R = []
    
    # for each line feature
    for i in range(len(T)): 
        
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
            
            # Get the Cartesian coordinate of point j of line i
            x_h = L[i][j][0]
            y_h = L[i][j][1]
            
            # Get the polar coordinate of point j of line i
            p_pc = cart2pol(x_h, y_h)
            d = p_pc[0]
            phi = p_pc[1]        
            
            # Entries of the Jacobian A
            A21 = ((y_bar - y_h) * (Sy_2 - Sx_2) + (2 * Sxy * (x_bar - x_h))) / (math.pow(Sy_2 - Sx_2,2) + 4 * math.pow(Sxy,2))
            A11 = (math.cos(alpha_) / n) - A21 * (x_bar * math.sin(alpha_) - y_bar * math.cos(alpha_))
            A22 = ((x_bar - x_h) * (Sy_2 - Sx_2) + (2 * Sxy * (y_bar - y_h))) / (math.pow(Sy_2 - Sx_2,2) + 4 * math.pow(Sxy,2))
            A12 = (math.sin(alpha_) / n) - A22 * (x_bar * math.sin(alpha_) - y_bar * math.cos(alpha_))
            
            # Jacobian A
            A_ = np.array([[A11,A12],
                           [A21,A22]])
            
            # Jacobian B
            B_ = np.array([[math.cos(phi), -d * math.sin(phi)],
                           [math.sin(phi), d * math.cos(phi)]])
            
            # Jacobian D
            # D should be estimated from the real LiDAR
            D_ = np.array([[0.01**2,0], 
                           [0,0]])
            
            # Calculate the line covaraince
            R_ = R_ + np.dot(A_, B_).dot(D_).dot(B_.T).dot(A_.T)
            
        R.append(R_)
    
    return R


# Calculate innovation covariance for confirmed landmarks
def get_innovation_covariance(j_H, sigmaxx_bar, sigmaxm_bar, sigmamm_bar, R, i, number_of_landmarks):
    
    S = []
    
    # for all landmarks
    for j in range(number_of_landmarks):
        
        # Build smaller sigma_bar than actual full covaraince list, for efficiency
        sigma_bar_1 = np.concatenate((sigmaxx_bar, sigmaxm_bar[:,(2*j):(2*j+2)]), axis = 1)
        sigma_bar_2 = np.concatenate((sigmaxm_bar[:,(2*j):(2*j+2)].T, sigmamm_bar[(2*j):(2*j+2),(2*j):(2*j+2)]), axis = 1)
        sigma_bar_small = np.concatenate((sigma_bar_1, sigma_bar_2), axis = 0)
        
        # Get innovation covariance
        S_ = np.dot(j_H[j], sigma_bar_small).dot(j_H[j].T) + R[i]
        
        S.append(S_)
    
    return S

# Calculate innovation covariance for tentative landmarks
def get_tentinnovation_covariance(j_H, sigmaxx_bar, sigmamm_bar, R, i, number_of_landmarkstent):
    
    S = []
    
    # for all landmarks
    for j in range(number_of_landmarkstent):
        
        # Build smaller sigma_bar than actual full covaraince list, for efficiency
        sigma_bar_1 = np.concatenate((sigmaxx_bar, np.zeros((3,2))), axis = 1)
        sigma_bar_2 = np.concatenate((np.zeros((2,3)), sigmamm_bar[:,(2*j):(2*j+2)]), axis = 1)
        sigma_bar_small = np.concatenate((sigma_bar_1, sigma_bar_2), axis = 0)
        
        # Get innovation covariance
        S_ = np.dot(j_H[j], sigma_bar_small).dot(j_H[j].T) + R[i]

        S.append(S_)
    
    return S


# Calculates Mahalanobis distance and nearest neighbour value to determine the correspondence of the feature
def get_correspondence(z_i, z_hat, S, assigned_landmarks):
    
    flag = 0
    correspondence = -1
    number_of_landmarks = len(z_hat)
    
    Mahalanobis_distance = np.empty(number_of_landmarks)
    nearest_neighbour = np.empty(number_of_landmarks)
    
    # variable to store minimum nearest neighbour value
    minimum = np.inf
    # variable to store minimum Mahalanobis distance
    min_Mahal = 10e6
    
    # validation gate threshold
    threshold = 6
    # angle threshold to handle angular nonlinearity at pi
    ang_thresh = 0.08
    
    # for all landmarks
    for i in range(number_of_landmarks):
        
        # Calculate innovation
        if z_i[1] < -np.pi + ang_thresh or z_hat[i,1] < -np.pi + ang_thresh:
            delta = np.abs(z_i) - np.abs(z_hat[i])
        else:
            delta = z_i - z_hat[i]
            
        inverse_S = np.linalg.inv(S[i])
        
        # Calculate Mahalobis distances from line feature z_i to all landmarks
        Mahalanobis_distance[i] = np.dot(delta.T,inverse_S).dot(delta)  
        
        # Apply validation gate (rejecting possible outlier correspondence)
        if Mahalanobis_distance[i] > threshold:
            
            print("Landmark ", i, ": Too large Mahalanobis distance: ", Mahalanobis_distance[i])
            continue
 
        # Calculate nearest neighbour values from line feature z_i to all landmarks in validation gate
        nearest_neighbour[i] = Mahalanobis_distance[i] + np.log(np.linalg.det(S[i]))
    
        # Find the correspondence that gives the minimum nearest_neighbour value
        if nearest_neighbour[i] < minimum:
            
            for k in range(len(assigned_landmarks)):
                
                if i == assigned_landmarks[k]:
                    
                    flag = 1
                    break
            
            if flag == 0:
                
                minimum = nearest_neighbour[i]
                correspondence = i
                
            flag = 0
        
        print("Landmark ", i, "-> ", "M: ", Mahalanobis_distance[i], " NN: ", nearest_neighbour[i], " C: ", correspondence)
    
    print("Final: ", "NN: ", minimum, " C: ", correspondence)
    
    if np.any(Mahalanobis_distance):
        min_Mahal = np.amin(Mahalanobis_distance)
    
    return correspondence, min_Mahal


# Calculate the Jacobians of the inverse observatipon model (wrt pose and measurement)
def jacobian_F(x_bar, psi, k):
    
    x = x_bar[0]
    y = x_bar[1]
    
    j_Fx = np.array([[math.cos(psi), math.sin(psi), -x*math.sin(psi) + y*math.cos(psi)],
                     [0, 0, 1]])
    j_Fw = np.array([[math.pow(-1,k + 1), -x*math.sin(psi) + y*math.cos(psi)],
                     [0, 1]])
    
    return j_Fx, j_Fw


# Add landamrk to the confirmed map  
def new_landmark(sigmaxx_bar, sigmaxm_bar, sigmamm_bar, R, j_Fx, j_Fw):
    
    # Calculate submatirces that should be appended to the current covaraince
    sigmanewnew_bar = np.dot(j_Fx, sigmaxx_bar).dot(j_Fx.T) + np.dot(j_Fw, R).dot(j_Fw.T)
    sigmaxall_bar = np.concatenate((sigmaxx_bar, sigmaxm_bar), axis = 1)
    sigmanewall_bar = np.dot(j_Fx, sigmaxall_bar)
    
    # Build sigmamm_bar after landmark addition
    sigmaxm_bar = np.concatenate((sigmaxm_bar, sigmanewall_bar[:,0:3].T), axis = 1)
    sigmamm_bar_1 = np.concatenate((sigmamm_bar, sigmanewall_bar[:,3:].T), axis = 1)
    sigmamm_bar_2 = np.concatenate((sigmanewall_bar[:,3:], sigmanewnew_bar), axis = 1)
    sigmamm_bar = np.concatenate((sigmamm_bar_1, sigmamm_bar_2), axis = 0)
    
    return sigmaxm_bar, sigmamm_bar


# Add landamrk to the tentative map  
def new_tentlandmark(sigmaxx_bar, sigmamm_bar, R, j_Fx):
    
    # Only ca;cu;ate landmark's variance and append it to the matrix
    sigmanewnew_bar = np.dot(j_Fx, sigmaxx_bar).dot(j_Fx.T) + np.dot(j_Fw, R).dot(j_Fw.T)
    sigmamm_bar = np.concatenate((sigmamm_bar, sigmanewnew_bar), axis = 1)
    
    return sigmamm_bar


# Kalman gain
def get_Kalman_gain(sigmaxx_bar, sigmaxm_bar, sigmamm_bar, j_H, S, i):
    
    # Build smaller sigma_bar than actual full covaraince list, for efficiency
    sigma_bar_1 = np.concatenate((sigmaxx_bar, sigmaxm_bar[:,(2*i):(2*i+2)]), axis = 1)
    sigma_bar_2 = np.concatenate((sigmaxm_bar.T, sigmamm_bar[:,(2*i):(2*i+2)]), axis = 1)
    sigma_bar_small = np.concatenate((sigma_bar_1, sigma_bar_2), axis = 0)
    
    # Compute Kalman gain
    Kalman_gain = np.dot(sigma_bar_small,j_H.T).dot(np.linalg.inv(S))
    
    return Kalman_gain


# Kalman covariance update  
def KF_update_covar(sigmaxx_bar, sigmaxm_bar, sigmamm_bar, j_H, K, i):
    
    # Build full covariance matrix from submatrices
    sigma_bar_1 = np.concatenate((sigmaxx_bar, sigmaxm_bar), axis = 1)
    sigma_bar_2 = np.concatenate((sigmaxm_bar.T, sigmamm_bar), axis = 1)
    sigma_bar = np.concatenate((sigma_bar_1, sigma_bar_2), axis = 0)
    
    # Build smaller sigma_bar than actual full covaraince list, for efficiency
    sigma_bar_small_1 = np.concatenate((sigmaxx_bar, sigmaxm_bar), axis = 1)
    sigma_bar_small_2 = np.concatenate((sigmaxm_bar[:,(2*i):(2*i+2)].T, sigmamm_bar[:,(2*i):(2*i+2)].T), axis = 1)
    sigma_bar_small = np.concatenate((sigma_bar_small_1, sigma_bar_small_2), axis = 0)
    
    # Compute full updated covariance
    sigma_bar = sigma_bar - np.dot(K,j_H).dot(sigma_bar_small)
    
    # Decompose sigma_bar in the submatirx blocks
    sigmaxx_bar = sigma_bar[0:3,0:3]
    sigmaxm_bar = sigma_bar[0:3,3:]
    sigmamm_bar = sigma_bar[3:,3:]
    
    return sigmaxx_bar, sigmaxm_bar, sigmamm_bar


# Remove landmarks from the tentative list
def delete_tent_landmark(g_mtent_pc_flat, g_mtent_pc, sigmamtentmtent_bar, number_of_tentassoc, number_of_iteration_since_found, corresp):
    
    # Delete from tentative map vector
    g_mtent_pc_flat = np.delete(g_mtent_pc_flat, [2*corresp, 2*corresp + 1], 0)
    g_mtent_pc = np.delete(g_mtent_pc, corresp, 0)
    
    # Delete from vecctor of tentative covariances 
    sigmamtentmtent_bar = np.delete(sigmamtentmtent_bar, [2*corresp, 2*corresp + 1], 1) 
    
    # Delete from reliability indicator vectors
    number_of_tentassoc = np.delete(number_of_tentassoc, corresp)
    number_of_iteration_since_found = np.delete(number_of_iteration_since_found, corresp)
    
    return g_mtent_pc_flat, g_mtent_pc, sigmamtentmtent_bar, number_of_tentassoc, number_of_iteration_since_found


# Plot line landmarks and their covariance ellipse on top of the real map
def plot_landmarks(g_m_xy, g_m_xy_true, g_m_mid, sigmamm):
    
    for k in range(len(g_m_xy)):
        euclidian_dist = np.zeros(((len(g_m_xy_true))))
        for i in range (len(g_m_xy_true)):
            euclidian_dist[i] = (g_m_xy_true[i,0] - g_m_xy[k,0])**2 + (g_m_xy_true[i,1] - g_m_xy[k,1])**2 
        j = np.argmin(euclidian_dist)
        landmarkx = g_m_xy[k,0]
        landmarky = g_m_xy[k,1]
        if j % 2 == 0:
            if j == 0 or j == 2:
                seg_leg = 13
            else:
                seg_leg = 0.4
            landmarkx = g_m_mid_true[j,0]
            plt.plot([landmarkx - seg_leg/2, landmarkx + seg_leg/2],[landmarky, landmarky], linestyle = '-', color = 'dodgerblue', lw = 6, zorder = 8)
        else:
            if j == 1 or j == 3:
                seg_leg = 6.18*1.3
            else:
                seg_leg = 0.4
            landmarky = g_m_mid_true[j,1]
            plt.plot([landmarkx, landmarkx],[landmarky - seg_leg/2, landmarky + seg_leg/2], linestyle = '-', color = 'dodgerblue', lw = 6, zorder = 8)     
        sigma_ = sigmamm[k:k+2,k:k+2]
        lambda_, v = np.linalg.eig(sigma_)
        lambda_ = np.sqrt(lambda_)
        m_ellipse = Ellipse((landmarkx, landmarky), width=lambda_[0]*5*2, height=lambda_[1]*5*2, angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), zorder = 9, ec = 'k', fc = 'lightcoral', lw = 2)
        plt.gca().add_artist(m_ellipse)


# Plot error ellipse of the pose
def plot_position_error_ellipse(x, sigmaxx):
    
    sigma_ = sigmaxx[0:2,0:2]
    lambda_, v = np.linalg.eig(sigma_)
    lambda_ = np.sqrt(lambda_)
    x_ellipse = Ellipse((x[0], x[1]), width=lambda_[0]*5*2, height=lambda_[1]*5*2, angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), zorder = 1, ec = 'k', fc = 'gold', lw = 2)
    plt.gca().add_artist(x_ellipse)

    
# plot time execution
def plot_landmark_covar_det(detSigma, number_of_iterations, number_of_landmarks):
    
    plt.figure(2)
    det_max = np.amax(detSigma)
    fig = plt.figure(figsize=(22, 10))
    for i in range(0,len(detSigma)):
        plt.step(np.arange(0, number_of_iterations), detSigma[i], where='post')
    ax = plt.gca()
    plt.ylim([0,1.1*det_max])
    plt.xticks(np.arange(0,226,25), fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.grid(color='black', linestyle='-', linewidth=0.25, alpha = 0.2, zorder = 1)
    plt.xlabel('Iteration Number', rotation = 0, fontsize = 35)
    plt.ylabel('$\det(\Sigma_{t,m_jm_j})$', rotation = 90, labelpad = 25, fontsize = 35)
    ax.yaxis.offsetText.set_fontsize(35)
    fig.tight_layout()
    plt.savefig('SLAM_det.svg')
    

# In[4]:

# Define constants

pose_shape = np.array([[-0.1,0.1], [-0.1,-0.1], [0.2,0]]) # triangle for plotting

scale = 1 # scale factor for time step and control
DT = 1/scale # time step [s] 

# pose covariance  matrix
sigmaxx = np.diag([0, 0, 0])**2 
# pose with map covariance
sigmaxm = np.empty(shape = [3, 0])
# map with map covariance
sigmamm = np.empty(shape = [0, 0])
# full tentative map with full tentative map covariance
sigmamtentmtent = np.empty(shape = [2, 0])

# define control input vector u
v = 1/4
w = [1, 2, 0.3, 3, -1, 1.5, -1.4, -1, 0.1, -1.3, 2, 2, 1.5, 0.2, -0.5, -1, -0.1, 1, 2.5, -2, 0.5, 2, 3.55, -0.95, -0.95, -0.65, -0.05, -2.5, 1.85, -0.4, -0.9, -0.1, -0.3, 1.45, -0.1, 0.9, 1, 0.4, 1, -3.25, 0.2, 0.7, 0.4, 2]
w = [x/4 for x in w]
u_id = 0

# process covariance
alpha_1 = 0.1
alpha_2 = 0.1
alpha_3 = 0.1
alpha_4 = 0.1
alpha_5 = 0.1
alpha_6 = 0.1
#Q = np.diag([alpha_1 * v**2 + alpha_2 * w**2, alpha_3 * v**2 + alpha_4 * w**2, alpha_5 * v**2 + alpha_6 * w**2])**2 
Q = (np.diag([0.05, 0.04, 0.02])/4)**2 

# define pose vector x
x = set_x(1.5,2,-np.pi/2) # (x,y,theta)
# define map vector
g_m_pc = np.empty(shape = [0, 2]); # 2 columns
g_m_pc_flat = np.empty(shape = [0, 1]);
# define tentative map vector
g_mtent_pc = np.empty(shape = [0, 2]); # 2 columns
g_mtent_pc_flat = np.empty(shape = [0, 1]);

# true pose and npiseless pose vector (for performance evaluation)
x_true = set_x(1.5,2,-np.pi/2)
x_odom = set_x(1.5,2,-np.pi/2)

# load the map of the environment: global point cloud in polar coordinates and polar/Cartesian coordinates of the line landmarks in global frame
g_point_cloud_pc, g_m_pc_true, g_m_xy_true, g_m_mid_true = set_map()

# number_of_true_landmarks 
true_number_of_landmarks = 40
# initial number of landmarks in the map is 0 
number_of_landmarks = 0
# initial number of tentative landmarks in the map is 0
number_of_landmarkstent = 0

# figure number of the animation
figure = 1

# global point cloud of the environment in cartesian coordinates
g_point_cloud_xy = pol2cart(g_point_cloud_pc[:,0], g_point_cloud_pc[:,1])

# number of iterations to run the algorithm
number_of_iterations = int(5*len(w)*scale)

# lists with zero entries to store robot's estimated and ground truth trajectories
trajectory_estimated = np.zeros((number_of_iterations + 1, 3))
trajectory_true = np.zeros((number_of_iterations + 1, 2))
trajectory_odom = np.zeros((number_of_iterations + 1, 2))

# initial position belief
trajectory_estimated[0,0] = x[0]
trajectory_estimated[0,1] = x[1]
trajectory_estimated[0,2] = x[2]

# initial true position
trajectory_true[0,0] = x_true[0]
trajectory_true[0,1] = x_true[1]

# initial noiseless position
trajectory_odom[0,0] = x_odom[0]
trajectory_odom[0,1] = x_odom[1]

# define parameters for landmark reliability
number_of_tentassoc = np.empty(shape = [1, 0]) # a in Algortihm 3
number_of_iteration_since_found = np.empty(shape = [1, 0]) # A in Algorithm 3

# flag that indicates when the first tentative landmark is found
first_tent_added = 0

# use d to only show a part of the trajectory
tr = 0

# excution time empty vector
exec_time = np.zeros((number_of_iterations))

# determinant of landmarks empty vector
detSigma = np.zeros((100, number_of_iterations)) - 1 

# start the algorithm
for j in range(number_of_iterations):
    
    if j % (5*scale) == 0:
        # set control input vector u
        u = set_u(v,w[u_id])
        u_id = u_id + 1
    
    start = time.time()
    
    # current determinats are the previous ones 
    for kk in range(0, number_of_landmarks):
            detSigma[kk,j] = detSigma[kk,j-1]
    
    print("Iteration:", j)

    # figure settings for the animation
    fig = plt.figure(figsize=(22,13.6))
    plotscale = 1.3
    ax = plt.gca()
    ax.set_xlim(-5.5 * plotscale, 5.5 * plotscale)
    ax.set_ylim(-3.75 * plotscale, 3.75 * plotscale)
    plt.xticks(np.arange(-6,6.1,2), fontsize = 35)
    plt.yticks(np.arange(-4,4.1,2), fontsize = 35)
    plt.scatter(1.5, 2, s = 80, color = 'm', zorder = 8)
    plt.xlabel('$x_{\mathcal{G}}$ [m]', rotation=0,fontsize = 35)
    plt.ylabel('$y_{\mathcal{G}}$ [m]', rotation=90, labelpad=10,fontsize = 35)
    plt.grid(color='black', linestyle='-', linewidth=0.25, alpha = 0.2, zorder = 1)
    
    # plot the global point cloud of the environment
    plt.plot(g_point_cloud_xy[0:1441,0], g_point_cloud_xy[0:1441,1], '.', color = 'dodgerblue', alpha = 0.03, zorder = 2, lw = 6)
    plt.plot(g_point_cloud_xy[1441:,0], g_point_cloud_xy[1441:,1], '.', color = 'dodgerblue', alpha = 0.006, zorder = 2, lw = 6)
        
    # Kalman prediction
    x_bar, sigmaxx_bar, sigmaxm_bar = KF_prediction(x, u, sigmaxx, sigmaxm, Q)
    sigmamm_bar = sigmamm
    # tentatove map unchanged
    sigmamtentmtent_bar = sigmamtentmtent
    
    # calculate x ground truth
    x_true = true_position(x_true, u)
    # saving it for the animation
    trajectory_true[j+1,0] = x_true[0]
    trajectory_true[j+1,1] = x_true[1]
    
    # calculate noiseless x ground truth
    x_odom = motion_model(x_odom, u)
    # saving it for the animation
    trajectory_odom[j+1,0] = x_odom[0]
    trajectory_odom[j+1,1] = x_odom[1]
    
    # plot true and noiseless robots as triangles
    pose_shape_rotated = rotate(pose_shape, (0,0), angle=x_true[2])
    t1 = plt.Polygon(pose_shape_rotated + x_true[0:2], color='lime', zorder = 6)
    plt.gca().add_patch(t1)
    pose_shape_rotated = rotate(pose_shape, (0,0), angle=x_odom[2])
    t3 = plt.Polygon(pose_shape_rotated + x_odom[0:2], color='C1', zorder = 5)
    plt.gca().add_patch(t3)
    
    # get LiDAR measurements from the body frame in polar and Cartesian coordinates
    l_point_cloud_pc = get_lidar_scan(g_point_cloud_pc, x_true, 1)
    l_point_cloud_xy = pol2cart(l_point_cloud_pc[:,0], l_point_cloud_pc[:,1])
    
    # show only the last few poses
    if j > 29:
        tr = j - 29
    
    # if no feature is detected, next iteration follows
    if len(l_point_cloud_pc) == 0:
            
        # update state and covariance
        x = x_bar
        sigmaxx = sigmaxx_bar
        sigmaxm = sigmaxm_bar
        sigmamm = sigmamm_bar
        sigmamtentmtent = sigmamtentmtent_bar
        
        # plot error ellipse for robot position
        plot_position_error_ellipse(x, sigmaxx)  
        
        # plot true, estimated and tentative landmarks
        g_m_xy = pol2cart(g_m_pc[:,0], g_m_pc[:,1])
        plt.scatter(g_m_xy[:,0], g_m_xy[:,1], 15, 'orange', zorder = 10) 
        plot_landmarks(g_m_xy, g_m_xy_true, g_m_mid_true, sigmamm)
        g_mtent_xy = pol2cart(g_mtent_pc[:,0], g_mtent_pc[:,1])
        plt.scatter(g_mtent_xy[:,0], g_mtent_xy[:,1], 15, 'b', zorder = 11)
        
        # saving position for the animation
        trajectory_estimated[j+1,0] = x[0]
        trajectory_estimated[j+1,1] = x[1]
        trajectory_estimated[j+1,2] = x[2]
        
        # plot estimated robot as triangle
        pose_shape_rotated = rotate(pose_shape, (0,0), angle=x[2])
        t2 = plt.Polygon(pose_shape_rotated + x[0:2], color='m', zorder = 7)
        plt.gca().add_patch(t2)
        
        # plotting trajectories
        plt.plot(trajectory_estimated[tr:j+2,0],trajectory_estimated[tr:j+2,1], 'm', linestyle = '--', zorder = 7, label = 'EKF', lw = 6)
        plt.plot(trajectory_true[tr:j+2,0],trajectory_true[tr:j+2,1],'lime', zorder = 6, label = 'True', lw = 6)
        plt.plot(trajectory_odom[tr:j+2,0],trajectory_odom[tr:j+2,1],'C1', zorder = 5, label = 'Motion model', lw = 6)
        
        plt.legend(mode = 'expand', loc = 'upper right', ncol = 5, fontsize = 30)
        fig.tight_layout(pad = 0)
        
        # saving figures for animation
        my_path = os.getcwd() + "/SLAM figures/"
        my_file = str(figure) + '.png'
        plt.savefig(os.path.join(my_path, my_file))
        figure = figure + 1 
        plt.close()
        
        # Compute time elapsed
        exec_time[j] = time.time() - start
            
        continue 
    
    # include the noisy lidar measurements in the animation figures
    l_point_cloud_xy_rotated = rotate(l_point_cloud_xy, (0,0), angle=x_true[2])
    plt.scatter(l_point_cloud_xy_rotated[:,0]+x_true[0],l_point_cloud_xy_rotated[:,1] + x_true[1], color = 'red', zorder = 3, label = 'LiDAR', s = 100)

    # get local map of the line features in polar coordinates (rho, alpha) that are extracted from LiDAR scans
    l_feature_m_pc, T, L = extract_features(l_point_cloud_pc, x_true, x_bar) 
    # T contains the intermediate line feature model parameters
    # L contains the cartesian coordinates of all points belonging to line features

    # compute line feature covariances
    alpha = l_feature_m_pc[:,1]
    R = estimate_line_feature_covariance(T, L, alpha)
    
    # empty array to hold the landmarks that are assigned to features
    assigned_landmarks = np.empty(shape = [1])
    # empty array to hold the tentative landmarks that are assigned to features
    assigned_landmarkstent = np.empty(shape = [1])
    
    number_of_features = len(R)
    
    # increment iteration numbers since found for all tentative landmarks
    if first_tent_added == 1:
        number_of_iteration_since_found += 1
        
        # delete tentative landmarks that haven't been matched for 15 iterations
        kk = len(number_of_iteration_since_found) - 1
        while kk >= 0:
        
            if number_of_iteration_since_found[kk] == 15:
                # delete the tentative landmark
                g_mtent_pc_flat, g_mtent_pc, sigmamtentmtent_bar, number_of_tentassoc, number_of_iteration_since_found = delete_tent_landmark(g_mtent_pc_flat, g_mtent_pc, sigmamtentmtent_bar, number_of_tentassoc, number_of_iteration_since_found, kk)
                number_of_landmarkstent -= 1
            kk -= 1 
        
    # for all features
    for i in range(number_of_features):
        
        # get polar coordinates of landmarks
        g_m_xy = pol2cart(g_m_pc[:,0], g_m_pc[:,1])
        
        # get expected landmark measurements
        z_hat, k = measurement_model(x_bar, g_m_pc)
        # converting to Cartesian coordinates
        z_hat_xy = pol2cart(z_hat[:,0],z_hat[:,1]+x_bar[2])
        
        # Jacobian of the measurement model
        psi = g_m_pc[:,1]
        j_H = jacobian_H(x_bar, psi, k)

        # calculate innovation covariances between i'th line feature and all landmarks in the map
        S = get_innovation_covariance(j_H, sigmaxx_bar, sigmaxm_bar, sigmamm_bar, R, i, number_of_landmarks)
                  
        # get i'th line feature local frame parameters (rho,alpha)
        z_i = l_feature_m_pc[i]
        
        print("feature: ",i)
        print("Assigned Landmarks:")
        for m in range(1,len(assigned_landmarks)):
            print(assigned_landmarks[m])
            
        # find which landmark the i'th feature corresponds to
        correspondence, min_Mahal = get_correspondence(z_i, z_hat, S, assigned_landmarks)
        
        #if min_Mahal > 6 and min_Mahal < 200:
            #continue
        
        # if no correspondence has been found, don't compute the Kalman gain and don't update x_bar and sigma_bar
        # it is time to check the tentative list
        if correspondence == -1:
            
            # get Cartesian coordinates of tentative landmark estimates
            g_mtent_xy = pol2cart(g_mtent_pc[:,0], g_mtent_pc[:,1])
            
            # get expected tentative landmark measurements
            z_hat, k = measurement_model(x_bar, g_mtent_pc)
            # converting to Cartesian coordinates
            z_hat_xy = pol2cart(z_hat[:,0],z_hat[:,1]+x_bar[2])
               
            # Jacobian of the measurement model
            psi = g_mtent_pc[:,1]
            j_H = jacobian_H(x_bar, psi, k)
            
            # calculate innovation covariances between i'th line feature and all landmarks in the tentative map  
            S = get_tentinnovation_covariance(j_H, sigmaxx_bar, sigmamtentmtent_bar, R, i, number_of_landmarkstent)
    
            print("Assigned Tentative Landmarks:")
            for m in range(1,len(assigned_landmarkstent)):
                print(assigned_landmarkstent[m])
                
            # find which tentative landmark i'th feature corresponds to
            correspondence, min_Mahal = get_correspondence(z_i, z_hat, S, assigned_landmarkstent)
            
            #if min_Mahal > 6 and min_Mahal < 200:
                #continue
            
            # if no correspondence has been found, the feature should be added to the tentative list
            if correspondence == -1:
                                
                # use inverse measurement model to get new tentative landmark from the feature
                rnew, psinew, k2 = inverse_measurement_model(x_bar, z_i)
                
                # expand the tentative map vector
                g_mtent_pc_flat = np.append(g_mtent_pc_flat, [[rnew],[psinew]], axis = 0)
                g_mtent_pc = np.append(g_mtent_pc, g_mtent_pc_flat[-2:].T, axis = 0)
                
                # Jacobian of inverse measuremnt model
                j_Fx, j_Fw = jacobian_F(x_bar, psinew, k2)
                
                # add the tentative landmark to the confirmed landmark list
                sigmamtentmtent_bar = new_tentlandmark(sigmaxx_bar, sigmamtentmtent_bar, R[i], j_Fx)
                
                # expand reliability vectors
                number_of_landmarkstent += 1
                number_of_tentassoc = np.append(number_of_tentassoc, [0])
                number_of_iteration_since_found = np.append(number_of_iteration_since_found, [0])
                
                if first_tent_added == 0:
                   first_tent_added = 1 
            
                continue
            
            # include the recent correspondence in the list
            assigned_landmarkstent = np.append(assigned_landmarkstent, [correspondence])
            
            number_of_tentassoc[correspondence] += 1
            
            # check  if a tentative landmdark should be moved to the confirmed list (3 successful associations)    
            if number_of_tentassoc[correspondence] == 3:
                
                # delete the tentative landmark
                g_mtent_pc_flat, g_mtent_pc, sigmamtentmtent_bar, number_of_tentassoc, number_of_iteration_since_found = delete_tent_landmark(g_mtent_pc_flat, g_mtent_pc, sigmamtentmtent_bar, number_of_tentassoc, number_of_iteration_since_found, correspondence)
                assigned_landmarkstent = np.delete(assigned_landmarkstent, np.where(assigned_landmarkstent == correspondence))
                number_of_landmarkstent -= 1
                
                # use inverse measurement model to get new landmark from the feature
                rnew, psinew, k2 = inverse_measurement_model(x_bar, z_i)
                
                # expand the map vector
                g_m_pc_flat = np.append(g_m_pc_flat, [[rnew], [psinew]], axis = 0)
                g_m_pc = np.append(g_m_pc, g_m_pc_flat[-2:].T, axis = 0)
                
                # Jacobian of inverse measuremtn model
                j_Fx, j_Fw = jacobian_F(x_bar, psinew, k2)
                
                # add the tentative landmark to the confirmed landmark list
                sigmaxm_bar, sigmamm_bar = new_landmark(sigmaxx_bar, sigmaxm_bar, sigmamm_bar, R[i], j_Fx, j_Fw)
                
                # calculate its initial determinant
                ddd =  sigmamm_bar[(len(sigmamm_bar) - 2):len(sigmamm_bar),(len(sigmamm_bar) - 2):len(sigmamm_bar)] + 0
                detSigma[number_of_landmarks,j] = np.linalg.det(ddd)
        
                number_of_landmarks += 1
                
            continue
    
        # include the recent correspondence in the list
        assigned_landmarks = np.append(assigned_landmarks, [correspondence])
        
        # compute the Kalman gain
        Kalman_gain = get_Kalman_gain(sigmaxx_bar, sigmaxm_bar, sigmamm_bar, j_H[correspondence], S[correspondence], correspondence)
        
        # calculate the innovation between the landmark and the corresponding feature
        innovation = z_i - z_hat[correspondence]
        
        # update x_bar, g_m_pc and sigma_bar with the knowledge obtained from detecting a landmark
        x_bar = x_bar + np.dot(Kalman_gain[0:3,:], innovation)
        x_bar[2] = fix_angle_one(x_bar[2])
        
        innovation = np.reshape(innovation, (-1, 1))
        
        # update map with kalman gain
        g_m_pc_flat = g_m_pc_flat + np.dot(Kalman_gain[3:,:], innovation)
        for q in range(0,np.size(g_m_pc_flat, 0),2):
            g_m_pc_flat[q + 1, 0] = fix_angle_one(g_m_pc_flat[q + 1, 0])    
            g_m_pc[int(q/2), 0] = g_m_pc_flat[q,0] 
            g_m_pc[int(q/2), 1] = g_m_pc_flat[q + 1,0]
        
        # update the covariance with the Kalman gain
        sigmaxx_bar, sigmaxm_bar, sigmamm_bar = KF_update_covar(sigmaxx_bar, sigmaxm_bar, sigmamm_bar, j_H[correspondence], Kalman_gain, correspondence)
        
        # compute updated determinants
        for kk in range(0, int(len(sigmamm_bar)/2)):
            detSigma[kk,j] = np.linalg.det(sigmamm_bar[2*kk:2*kk+2,2*kk:2*kk+2]) 
        
    print()
    
    # update state and covariance
    x = x_bar
    sigmaxx = sigmaxx_bar
    sigmaxm = sigmaxm_bar
    sigmamm = sigmamm_bar
    sigmamtentmtent = sigmamtentmtent_bar
    
    # plot error ellipse for robot position
    plot_position_error_ellipse(x, sigmaxx)  
    
    # plot true, estimated and tentative landmarks
    g_m_xy = pol2cart(g_m_pc[:,0], g_m_pc[:,1])
    plt.scatter(g_m_xy[:,0], g_m_xy[:,1], 15, 'orange', zorder = 10) 
    plot_landmarks(g_m_xy, g_m_xy_true, g_m_mid_true, sigmamm)
    g_mtent_xy = pol2cart(g_mtent_pc[:,0], g_mtent_pc[:,1])
    plt.scatter(g_mtent_xy[:,0], g_mtent_xy[:,1], 15, 'b', zorder = 11)
    
    # saving position for the animation
    trajectory_estimated[j+1,0] = x[0]
    trajectory_estimated[j+1,1] = x[1]
    trajectory_estimated[j+1,2] = x[2]
    
    # plot estimated robot as triangle
    pose_shape_rotated = rotate(pose_shape, (0,0), angle=x[2])
    t2 = plt.Polygon(pose_shape_rotated + x[0:2], color='m', zorder = 7)
    plt.gca().add_patch(t2)
    
    # plotting trajectories
    plt.plot(trajectory_estimated[tr:j+2,0],trajectory_estimated[tr:j+2,1], 'm', linestyle = '--', zorder = 7, label = 'EKF', lw = 6)
    plt.plot(trajectory_true[tr:j+2,0],trajectory_true[tr:j+2,1],'lime', zorder = 6, label = 'True', lw = 6)
    plt.plot(trajectory_odom[tr:j+2,0],trajectory_odom[tr:j+2,1],'C1', zorder = 5, label = 'Motion model', lw = 6)
    
    plt.legend(mode = 'expand', loc = 'upper right', ncol = 5, fontsize = 30)
    fig.tight_layout(pad = 0)
    
    # saving figures for animation
    my_path = os.getcwd() + "/SLAM figures/"
    my_file = str(figure) + '.png'
    plt.savefig(os.path.join(my_path, my_file))
    figure = figure + 1 
    plt.close()
    
    # Compute time elapsed
    exec_time[j] = time.time() - start

plot_landmark_covar_det(detSigma, number_of_iterations, number_of_landmarks)