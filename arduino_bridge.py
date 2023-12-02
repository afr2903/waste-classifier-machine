import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

while True:
    ser.write(b'1')
    time.sleep(3)
    ser.write(b'2')
    time.sleep(3)