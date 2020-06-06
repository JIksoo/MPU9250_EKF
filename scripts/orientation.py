#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from math import sqrt
import numpy as np
import time
from MPU9250 import IMU

class orientation:
    def __init__(self):
        self.imu = IMU(500, 4, 16)

        # Set rate time
        self.ref_time = time.time()
        self.dt = 0

        # get sensor bias
        self.gyroBias = rospy.get_param('/gyro_bias', None)
        self.accBias = rospy.get_param('/acc_bias', None)
        self.magBias = rospy.get_param('/mag_bias', None)
        self.magScale = rospy.get_param('/mag_scale', None)

        # get sensor Covariance Matrix
        gyroVariance = rospy.get_param('/gyro_variance', None)
        self.gyroCovariance_Matrix = np.matrix([[gyroVariance[0], 0, 0],
                                                [0, gyroVariance[1], 0],
                                                [0, 0, gyroVariance[2]]], dtype=np.float32)

        accVariance = rospy.get_param('/acc_variance', None)
        self.accCovariance_Matrix = np.matrix([[accVariance[0], 0, 0],
                                                [0, accVariance[1], 0],
                                                [0, 0, accVariance[2]]], dtype=np.float32)

        magVariance = rospy.get_param('/mag_variance', None)
        self.magCovariance_Matrix = np.matrix([[magVariance[0], 0, 0],
                                                [0, magVariance[1], 0],
                                                [0, 0, magVariance[2]]], dtype=np.float32)

        # Set gravitiy and magnetic field vector
        self.g = np.matrix([[0], [0], [0], [1]], dtype=np.float32)
        self.b = np.matrix([[0], [0], [0], [0]], dtype=np.float32)

        # Set Initial orientation and Covariance Matrix
        self.X = self.get_init_orientation()
        self.P = np.eye(4)

        # Set Validation Test Constants
        self.e_a = np.linalg.norm(self.g) + 0.2
        self.e_m = np.linalg.norm(self.b) + 100

        # Publish
        self.pose = PoseStamped()
        self.KF_pub = rospy.Publisher('/pose', PoseStamped, queue_size= 1)

    def get_imu_data(self):
        # Measure Time diviation
        T = time.time()
        self.dt = T - self.ref_time
        self.ref_time = T

        # Get Acc and Mag data
        self.imu.readRawIMU()

        gyro = [self.imu.gx - self.gyroBias[0], 
                self.imu.gy - self.gyroBias[1],
                self.imu.gz- self.gyroBias[2]]

        acc = [self.imu.ax - self.accBias[0], 
                self.imu.ay - self.accBias[1],
                self.imu.az- self.accBias[2]]

        mag = [self.imu.mx - self.magBias[0], 
                self.imu.my - self.magBias[1],
                self.imu.mz- self.magBias[2]]

        return gyro, acc, mag

    def get_init_orientation(self):
        gyro, acc, mag = self.get_imu_data()
        
        # Normalization acc        
        acc_norm = sqrt(sum([i**2 for i in acc]))
        acc = [i/acc_norm for i in acc]

        # Get q_acc
        if acc[2] >= 0:
            q_acc = np.matrix([[sqrt((acc[2]+1) / 2)],
                                [-acc[1] / sqrt(2*(acc[2]+1))],
                                [acc[0] / sqrt(2*(acc[2]+1))],
                                [0]], dtype=np.float32)
        else:
            q_acc = np.matrix([[-acc[1] / sqrt(2*(1 - acc[2]))],
                                [sqrt((1 - acc[2]) / 2)],
                                [0],
                                [acc[0] / sqrt(2*(1 - acc[2]))]], dtype=np.float32)

        # get q_mag
        l = quat_mult(quat_mult(inv_quat(q_acc), np.matrix([[0], [mag[0]], [mag[1]], [mag[2]]])), q_acc)
        b = sqrt(l[1,0]**2 + l[2,0]**2)

        self.b[1,0] = b     # get b_x
        self.b[3,0] = l[3,0]  # get b_z

        if l[1,0] >= 0:
            q_mag = np.matrix([[sqrt(0.5*(l[1,0]/b + 1))],
                                [0],
                                [0],
                                [l[2,0] / sqrt(2*b*(l[1,0] + b))]], dtype=np.float32)
        else:
            q_mag = np.matrix([[l[2,0] / sqrt(2*b*(b - l[1,0]))],
                                [0],
                                [0],
                                [sqrt(0.5*(1 - l[1,0]/b))]], dtype=np.float32)
                                
        q = quat_mult(q_acc, q_mag)
        
        return q

    def h(self, q):
        a = quat_mult(quat_mult(q, self.g), inv_quat(q))
        m = quat_mult(quat_mult(q, self.b), inv_quat(q))

        return np.concatenate((a[1:4],m[1:4]),axis=0)

    def Kalman(self):
        gyro, acc, mag = self.get_imu_data()



        # Predict
        F = np.eye(4) + 0.5 * self.dt * np.matrix([[0, gyro[0], gyro[1], gyro[2]],
                                                    [-gyro[0], 0, gyro[2], -gyro[1]],
                                                    [-gyro[1], -gyro[2], 0, gyro[0]],
                                                    [-gyro[2], gyro[1], -gyro[0], 0]], dtype=np.float64)

        Xp = F * self.X
        Xp /= np.linalg.norm(Xp)   # normalizatioon

        self.X = Xp

        I = np.matrix([[self.X[1,0], self.X[2,0], self.X[3,0]],
                        [-self.X[0,0], self.X[3,0], -self.X[2,0]],
                        [self.X[3,0], -self.X[0,0], -self.X[1,0]],
                        [-self.X[2,0], self.X[1,0], -self.X[0,0]]])

        Q = (self.dt**2)/4 * I * self.gyroCovariance_Matrix * I.T

        Pp = F * self.P * F.T + Q

        # Observasion
        H_a = np.matrix([[Xp[2,0], Xp[3,0], Xp[0,0], Xp[1,0]],
                        [-Xp[1,0], -Xp[0,0], Xp[3,0], Xp[2,0]],
                        [Xp[0,0], -Xp[1,0], -Xp[2,0], Xp[3,0]]])

        H_b1 = 1.0 / self.magScale[0] * np.matrix([self.b[1,0], self.b[3,0]]) * np.matrix([[Xp[0,0], Xp[1,0], -Xp[2,0], -Xp[3,0]],
                                                                            [Xp[2,0], Xp[3,0], Xp[0,0], Xp[1,0]]])

        H_b2 = 1.0 / self.magScale[1] * np.matrix([self.b[1,0], self.b[3,0]]) * np.matrix([[Xp[3,0], Xp[2,0], Xp[1,0], Xp[0,0]],
                                                                            [-Xp[1,0], -Xp[0,0], Xp[3,0], Xp[2,0]]])

        H_b3 = 1.0 / self.magScale[2] * np.matrix([self.b[1,0], self.b[3,0]]) * np.matrix([[-Xp[2,0], Xp[3,0], -Xp[0,0], Xp[1,0]],
                                                                            [Xp[0,0], -Xp[1,0], -Xp[2,0], Xp[3,0]]])

        H = np.concatenate((H_a, H_b1, H_b2, H_b3), axis=0)

        Z = np.matrix([[acc[0]],
                        [acc[1]],
                        [acc[2]],
                        [mag[0]],
                        [mag[1]],
                        [mag[2]]])

        y = Z - self.h(Xp)

        # Validation Test
        inf = 1000
        if sqrt(sum([i**2 for i in acc])) < self.e_a:
            R_a = np.concatenate((self.accCovariance_Matrix, np.zeros((3,3))), axis=1)
        else:
            R_a = np.matrix([[inf, 0, 0, 0, 0, 0],
                        [0, inf, 0, 0, 0, 0],
                        [0, 0, inf, 0, 0, 0]])

        if sqrt(sum([(mag[i] * self.magScale[i])**2 for i in range(3)])) < self.e_m:
            R_m = np.concatenate((np.zeros((3,3)), self.magCovariance_Matrix), axis=1)
        else:
            R_m = np.matrix([[0, 0, 0, inf, 0, 0],
                        [0, 0, 0, 0, inf, 0],
                        [0, 0, 0, 0, 0, inf]])

        R = np.concatenate((R_a, R_m), axis=0)

        K = Pp * H.T * np.linalg.inv(H*Pp*H.T + R)

        self.X = Xp + K * y
        self.X /= np.linalg.norm(self.X)
        self.P = (np.eye(4) - K*H) * Pp

    def publish(self):
        self.pose.header.stamp = rospy.Time.now()

        self.pose.header.frame_id = "base_link"

        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 0

        self.pose.pose.orientation.w = self.X[0,0]
        self.pose.pose.orientation.x = self.X[1,0]
        self.pose.pose.orientation.y = self.X[2,0]
        self.pose.pose.orientation.z = self.X[3,0]

        self.KF_pub.publish(self.pose)

def quat_mult(q1, q2):
        q = np.matrix([[0], [0], [0], [0]], dtype=np.float32)

        q[0,0] = q1[0,0]*q2[0,0] - q1[1,0]*q2[1,0] - q1[2,0]*q2[2,0] - q1[3,0]*q2[3,0]
        q[1,0] = q1[0,0]*q2[1,0] + q1[1,0]*q2[0,0] + q1[2,0]*q2[3,0] - q1[3,0]*q2[2,0]
        q[2,0] = q1[0,0]*q2[2,0] + q1[2,0]*q2[0,0] + q1[3,0]*q2[1,0] - q1[1,0]*q2[3,0] 
        q[3,0] = q1[0,0]*q2[3,0] + q1[3,0]*q2[0,0] + q1[1,0]*q2[2,0] - q1[2,0]*q2[1,0] 

        return q

def inv_quat(q):
    return np.matrix([[q[0,0]], [-q[1,0]], [-q[2,0]], [-q[3,0]]])

if __name__ == "__main__":
    rospy.init_node("Orientation_node")

    ori = orientation()

    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        ori.Kalman()

        ori.publish()

        rate.sleep()