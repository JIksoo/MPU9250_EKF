#!/usr/bin/env python

import rospy
import time
import matplotlib.pyplot as plt
from MPU9250 import IMU

class Calibration:
    def __init__(self):
        self.imu = IMU(500, 4, 16)

        self.magCalibratedData = [[],[],[]]

        self.gyroBias = [0, 0, 0]
        self.accBias = [0, 0, 0]
        self.magBias = [0, 0, 0]
        self.magScale = [0, 0, 0]

        self.gyroVariance = [0, 0, 0]
        self.accVariance = [0, 0, 0]
        self.magVariance = [0, 0, 0]

    def calibrateGyro(self, N):
        # Display message
        print("Calibrating gyro with " + str(N) + " points. Do not move!")

        gyroData = [[],[],[]]

        # Take N readings for each coordinate and add to itself
        for i in range(N):
            self.imu.readRawIMU()

            gyroData[0].append(self.imu.gx)
            gyroData[1].append(self.imu.gy)
            gyroData[2].append(self.imu.gz)

        # Find Bias and Variance
        for i in range(3):
            self.gyroBias[i] = sum(gyroData[i]) / N

            tmp = 0
            for j in gyroData[i]:
                tmp += (j - self.gyroBias[i])**2

            self.gyroVariance[i] = tmp / N

    def calibrateAcc(self, N):
        # Display message
        print("Calibrating accelerometer with " + str(N) + " points. Do not move!")

        accData = [[],[],[]]

        # Take N readings for each coordinate and add to itself
        for i in range(N):
            self.imu.readRawIMU()

            accData[0].append(self.imu.ax)
            accData[1].append(self.imu.ay)
            accData[2].append(self.imu.az - 1)

        # Find Bias and Variance
        for i in range(3):
            self.accBias[i] = sum(accData[i]) / N

            tmp = 0
            for j in accData[i]:
                tmp += (j - self.accBias[i])**2
                
            self.accVariance[i] = tmp / N

    def calibrateMag(self, N):
        self.varianceMag(100)

        print("Magnetometer Calibration Start")

        magData = [[],[],[]]

        # Local calibration variables
        magMin = [32767, 32767, 32767]
        magMax = [-32767, -32767, -32767]
        magTemp = [0, 0, 0]
        magChord = [0, 0, 0]

        # Take N readings of mag data
        for i in range(N):
            # Read fresh values and assign to magTemp
            self.imu.readRawIMU()
            magTemp = [self.imu.mx, self.imu.my, self.imu.mz]

            # Adjust the max and min points based off of current reading
            for j in range(3):
                magData[j].append(magTemp[j])
                if (magTemp[j] > magMax[j]):
                    magMax[j] = magTemp[j]
                if (magTemp[j] < magMin[j]):
                    magMin[j] = magTemp[j]

            # Display some info to the user
            if (i%50 == 0):
                print(str(100.0*i/N) + "%")

            # Small delay before next loop (data available every 10 ms or 100 Hz)
            time.sleep(0.05)

        # Get Hard iron and Soft iron
        self.magBias = [(magMax[i] + magMin[i])/2 for i in range(3)]

        magChord = [float(magMax[i] - magMin[i])/2 for i in range(3)]
        avgChord = sum(magChord) / 3.0
        self.magScale = [avgChord / magChord[i] for i in range(3)]

        # Get Calibrated Data
        for i in range(3):
            for j in magData[i]:
                self.magCalibratedData[i].append((j - self.magBias[i]) * self.magScale[i])

        print("Done")

    def varianceMag(self, N):
        magData = [[],[],[]]
        avg = [0, 0, 0]

        # Take N readings for each coordinate and add to itself
        for i in range(N):
            self.imu.readRawIMU()

            magData[0].append(self.imu.mx)
            magData[1].append(self.imu.my)
            magData[2].append(self.imu.mz)

        # Find Bias and Variance
        for i in range(3):
            avg[i] = sum(magData[i]) / N

            tmp = 0
            for j in magData[i]:
                tmp += (j - avg[i])**2
                
            self.magVariance[i] = tmp / N

    def calibratedMag_plot(self):
        # X-Y plane
        plt.figure(num=1)
        plt.title("X-Y")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.scatter(self.magCalibratedData[0], self.magCalibratedData[1])

        # Y-Z plane
        plt.figure(num=2)
        plt.title("Y-Z")
        plt.xlabel("Y-axis")
        plt.ylabel("Z-axis")
        plt.scatter(self.magCalibratedData[1], self.magCalibratedData[2])

        # Z-X plane
        plt.figure(num=3)
        plt.title("Z-X")
        plt.xlabel("Z-axis")
        plt.ylabel("X-axis")
        plt.scatter(self.magCalibratedData[2], self.magCalibratedData[0])

        plt.show()


if __name__ == "__main__":
    rospy.init_node("calibration_node")

    calibration = Calibration()

    calibration.calibrateGyro(100)
    calibration.calibrateAcc(100)
    calibration.calibrateMag(500)

    calibration.calibratedMag_plot()

    rospy.set_param("/gyro_bias", calibration.gyroBias)
    rospy.set_param("/gyro_variance", calibration.gyroVariance)
    rospy.set_param("/acc_bias", calibration.accBias)
    rospy.set_param("/acc_variance", calibration.accVariance)
    rospy.set_param("/mag_bias", calibration.magBias)
    rospy.set_param("/mag_scale", calibration.magScale)
    rospy.set_param("/mag_variance", calibration.magVariance)