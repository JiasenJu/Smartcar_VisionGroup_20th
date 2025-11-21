from pyb import LED #导入LED
from machine import Pin # 从machine导入Pin
import sensor
import time
import image
import os
import seekfree, pyb


# 显示屏接线
# IO接线方法
# 屏         openart
# GND       ---> GND
# VEE       ---> 3.3V
# SCL       ---> B0(SCLK)
# SDA/MISO  ---> B1(MISO)
# RESET     ---> B12
# DC        ---> B13
# CS        ---> B3
# BL        ---> B16 (背光控制)


#EXPOSURE_TIME = 1100   # 摄像头曝光值
sensor.reset() #初始化感光元件
sensor.set_pixformat(sensor.RGB565) #设置为RGB565色彩空间
sensor.set_framesize(sensor.QVGA) #设置图像的大小
sensor.set_auto_gain(False) #自动增益关闭
sensor.set_auto_whitebal(False) #自动白平衡关闭
sensor.set_auto_exposure(False, exposure_us = 1100) #自动曝光关闭
sensor.skip_frames(time = 100) #等待200ms使感光元件稳定


clock = time.clock() #读取帧率
#Light = pyb.LED(4) #定义补光灯

PC7 = Pin("C7",Pin.IN)

#pin10 = Pin("B0",0)#设置p_in为输入引脚，并开启上拉电阻
#pin10.init(Pin.IN, Pin.PULL_DOWN)

print("Please press the button to take a picture ！")
print("PC7 = %d" %(PC7.value()))

# 文件路径
file_path1 = "/sd/number/9/picture_i.txt"


# 尝试读取文件中的最后一个i的值
def read_last_i(file_path1):
    try:
        with open(file_path1, "r") as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()  # 获取最后一行并去除换行符
                return int(last_line)  # 返回最后一个i的值
    except OSError:
        pass  # 如果文件不存在，忽略异常
    return 0  # 如果文件不存在或没有内容，则返回0

# 将新的i值追加到文件中
def append_i(file_path1, i):
    try:
        with open(file_path1, "a+") as file:
            file.write(str(i) + "\n")
#            file.close()
            file.flush()  # 立即写入磁盘
    except OSError:
        pass  # 如果写入文件时发生错误，忽略异常

# 读取文件中的最后一个i的值
i = read_last_i(file_path1)
# 打印当前i的值
print("Current i:", i)

# 图片存储地址
folder_path = '/sd/number/'

#sensor.set_brightness(500)  # 根据需要调整亮度
sensor.skip_frames(time=2000)  # 跳过n张照片或者跳过time毫秒的帧数，等待传感器稳定
#sensor.set_windowing((320, 240))

key = 0
while(True):
    img = sensor.snapshot()

    key = 0
    if PC7.value() == 1:
        time.sleep_ms(20)
        key = 1
        time.sleep_ms(20)

    print("k = %d" %(key))

    if key == 1:
        # 获取图像
        print("NUM_picture = %d" %(i))
        img1 = img.copy() # 复制识别区域
        # 生成图像文件名
        filename = "9({}).jpg".format(i)

        # 拼接完整的文件路径
        file_path = folder_path + '9/' + filename
        # 保存图像
        img1.save(file_path)
        append_i(file_path1, i)
        i += 1  # 递增计数器
        key = 0

    print("k = %d" %(key))
    time.sleep_ms(600)

