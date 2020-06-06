#!/usr/bin/env python

import smbus
import time

PI = 3.141592


# gyro:  250, 500, 1000, 2000 [deg/s]
# acc: 2, 4, 7, 16 [g]
# mag: 14, 16 [bit]

class IMU:
    def __init__(self, gyro, acc, mag):
        # Class / object / constructor setup
        self.ax = None; self.ay = None; self.az = None
        self.gx = None; self.gy = None; self.gz = None
        self.mx = None; self.my = None; self.mz = None

        self.gyroScaleFactor, self.gyroHex = self.gyroSensitivity(gyro)
        self.accScaleFactor,  self.accHex  = self.accelerometerSensitivity(acc)
        self.magScaleFactor,  self.magHex  = self.magnetometerSensitivity(mag)

        self.magXcal = 0; self.magYcal = 0; self.magZcal = 0

        self.bus = smbus.SMBus(1)

        self.MPU9250_ADDRESS  = 0x68
        self.AK8963_ADDRESS   = 0x0C
        self.WHO_AM_I_MPU9250 = 0x75
        self.WHO_AM_I_AK8963  = 0x00

        self.AK8963_XOUT_L    = 0x03
        self.AK8963_CNTL      = 0x0A
        self.AK8963_CNTL2     = 0x0B
        self.USER_CTRL        = 0x6A
        self.I2C_SLV0_DO      = 0x63

        self.AK8963_ASAX      = 0x10
        self.I2C_MST_CTRL     = 0x24
        self.I2C_SLV0_ADDR    = 0x25
        self.I2C_SLV0_REG     = 0x26
        self.I2C_SLV0_CTRL    = 0x27
        self.EXT_SENS_DATA_00 = 0x49
        self.GYRO_CONFIG      = 0x1B
        self.ACCEL_XOUT_H     = 0x3B
        self.PWR_MGMT_1       = 0x6B
        self.ACCEL_CONFIG     = 0x1C

        self.setUpIMU()
        self.setUpMAG()

    def gyroSensitivity(self, x):
        # Create dictionary with standard value of 500 deg/s
        return {
            250:  [131.0, 0x00],
            500:  [65.5,  0x08],
            1000: [32.8,  0x10],
            2000: [16.4,  0x18]
        }.get(x,  [65.5,  0x08])

    def accelerometerSensitivity(self, x):
        # Create dictionary with standard value of 4 g
        return {
            2:  [16384.0, 0x00],
            4:  [8192.0,  0x08],
            8:  [4096.0,  0x10],
            16: [2048.0,  0x18]
        }.get(x,[8192.0,  0x08])

    def magnetometerSensitivity(self, x):
        # Create dictionary with standard value of 16 bit
        return {
            14:  [10.0*4912.0/8190.0,  0x06],
            16:  [10.0*4912.0/32760.0, 0x16],
        }.get(x,[10.0*4912.0/32760.0,  0x16])

    def setUpIMU(self):
        # Check to see if there is a good connection with the MPU 9250
        try:
            whoAmI = self.bus.read_byte_data(self.MPU9250_ADDRESS, self.WHO_AM_I_MPU9250)
        except:
            print('whoAmI IMU read failed')
            return

        if (whoAmI == 0x71):
            # Connection is good! Activate/reset the IMU
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.PWR_MGMT_1, 0x00)

            # Configure the accelerometer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.ACCEL_CONFIG, self.accHex)

            # Configure the gyro
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.GYRO_CONFIG, self.gyroHex)

            # Display message to user
            print("MPU set up:")
            print('\tAccelerometer: ' + str(hex(self.accHex)) + ' ' + str(self.accScaleFactor))
            print('\tGyroscope: ' + str(hex(self.gyroHex)) + ' ' + str(self.gyroScaleFactor) + "\n")
        else:
            # Bad connection or something went wrong
            print("IMU WHO_AM_I was: " + hex(whoAmI) + ". Should have been " + hex(0x71))

    def setUpMAG(self):
        # Initialize connection with mag for a WHO_AM_I test
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.USER_CTRL, 0x20)                              # Enable I2C Master mode
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_MST_CTRL, 0x0D)                           # I2C configuration multi-master I2C 400KHz
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS | 0x80)    # Set the I2C slave address of AK8963 and set for read.
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.WHO_AM_I_AK8963)           # I2C slave 0 register address from where to begin data transfer
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x81)                          # Enable I2C and transfer 1 byte
        time.sleep(0.05)

        # Check to see if there is a good connection with the mag
        try:
            whoAmI = self.bus.read_byte_data(self.MPU9250_ADDRESS, self.EXT_SENS_DATA_00)
        except:
            print('whoAmI MAG read failed')
            return

        if (whoAmI == 0x48):
            # Connection is good! Begin the true initialization
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS)     # Set the I2C slave address of AK8963 and set for write.
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_CNTL2)        # I2C slave 0 register address from where to begin data transfer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_DO, 0x01)                      # Reset AK8963
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x81)                    # Enable I2C and write 1 byte
            time.sleep(0.05)

            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS)     # Set the I2C slave address of AK8963 and set for write.
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_CNTL)         # I2C slave 0 register address from where to begin data transfer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_DO, 0x00)                      # Power down magnetometer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x81)                    # Enable I2C and transfer 1 byte
            time.sleep(0.05)

            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS)     # Set the I2C slave address of AK8963 and set for write.
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_CNTL)         # I2C slave 0 register address from where to begin data transfer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_DO, 0x0F)                      # Enter self-test mode
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x81)                    # Enable I2C and write 1 byte
            time.sleep(0.05)

            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS | 0x80)   # Set the I2C slave address of AK8963 and set for read.
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_ASAX)              # I2C slave 0 register address from where to begin data transfer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x83)                         # Enable I2C and read 3 bytes
            time.sleep(0.05)

            # Read the x, y, and z sensitivity adjustment value
            try:
                rawData = self.bus.read_i2c_block_data(self.MPU9250_ADDRESS, self.EXT_SENS_DATA_00, 3)
            except:
                print('Reading MAG x y z calibration values failed')
                return

            # Calculate sensitivity adjustment factor
            self.magXcal = float(rawData[0] - 128)/256.0 + 1.0
            self.magYcal = float(rawData[1] - 128)/256.0 + 1.0
            self.magZcal = float(rawData[2] - 128)/256.0 + 1.0

            # Flush the sysem
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS)     # Set the I2C slave address of AK8963 and set for write.
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_CNTL)         # I2C slave 0 register address from where to begin data transfer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_DO, 0x00)                      # Power down magnetometer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x81)                    # Enable I2C and write 1 byte

            # Configure the settings for the mag
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS)     # Set the I2C slave address of AK8963 and set for write.
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_CNTL)         # I2C slave 0 register address from where to begin data transfer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_DO, self.magHex)               # Set magnetometer for 14 or 16 bit continous 100 Hz sample rates
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x81)                    # Enable I2C and transfer 1 byte
            time.sleep(0.05)

            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS | 0x80)    # Set the I2C slave address of AK8t963 and set for read.
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_CNTL)               # I2C slave 0 register address from where to begin data transfer
            self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x81)                          # Enable I2C and transfer 1 byte
            time.sleep(0.05)

            # Display results to user
            print("MAG set up:")
            print("\tMagnetometer: " + hex(self.magHex) + " " + str(round(self.magScaleFactor,3)) + "\n")
        else:
            # Bad connection or something went wrong
            print("MAG WHO_AM_I was: " + hex(whoAmI) + ". Should have been " + hex(0x48))

    def eightBit2sixteenBit(self, l, h):
        # Shift the low and high byte into a 16 bit number
        val = (h << 8) + l

        # Make 16 bit unsigned value to signed value (0 to 65535) to (-32768 to +32767)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

    def readRawIMU(self):
        # Read accelerometer and gyro
        # Read 14 raw values [High Low] as temperature falls between the accelerometer and gyro registries
        try:
            rawData = self.bus.read_i2c_block_data(self.MPU9250_ADDRESS, self.ACCEL_XOUT_H, 14)
        except:
            print('Read raw IMU data failed')

        # Convert the raw values to something a little more useful (middle value is temperature)
        self.ax = self.eightBit2sixteenBit(rawData[1], rawData[0]) / self.accScaleFactor
        self.ay = self.eightBit2sixteenBit(rawData[3], rawData[2]) / self.accScaleFactor
        self.az = self.eightBit2sixteenBit(rawData[5], rawData[4]) / self.accScaleFactor

        self.gx = self.eightBit2sixteenBit(rawData[9], rawData[8]) / self.gyroScaleFactor * PI / 180
        self.gy = self.eightBit2sixteenBit(rawData[11], rawData[10]) / self.gyroScaleFactor * PI / 180
        self.gz = self.eightBit2sixteenBit(rawData[13], rawData[12]) / self.gyroScaleFactor * PI / 180

        # Read magnetometer
        # Prepare to request values
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_ADDR, self.AK8963_ADDRESS | 0x80)    # Set the I2C slave address of AK8963 and set for read.
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_REG, self.AK8963_XOUT_L)             # I2C slave 0 register address from where to begin data transfer
        self.bus.write_byte_data(self.MPU9250_ADDRESS, self.I2C_SLV0_CTRL, 0x87)                          # Enable I2C and read 7 bytes
        time.sleep(0.005)

        # Read 7 values [Low High] and one more byte (overflow check)
        try:
            rawData = self.bus.read_i2c_block_data(self.MPU9250_ADDRESS, self.EXT_SENS_DATA_00, 7)
        except:
            print('Read raw MAG data failed')

        # If overflow check passes convert the raw values to something a little more useful
        if not (rawData[6] & 0x08):
            self.mx = self.eightBit2sixteenBit(rawData[2], rawData[3]) * self.magScaleFactor * self.magYcal
            self.my = self.eightBit2sixteenBit(rawData[0], rawData[1]) * self.magScaleFactor * self.magXcal
            self.mz = -self.eightBit2sixteenBit(rawData[4], rawData[5]) * self.magScaleFactor * self.magZcal