######  本版本区分为47版整合数字和图片识别

## 初步实现四个方向数字识别

#不再局限于检测图片的矩形区域，而是将整个场景拍摄并保存为图像。
#对于每一类目标图片，我们既可以通过实际拍摄得到，也可以利用工
#具将数据集中的图片合成到不同的背景中，以此构建训练所需的数据
#集，随后进行训练。识别时只需要确保车模在箱子正前方，直接识别整幅图像。



    #A[输入图像] --> B{粗分类模型}
    #B -->|手写数字| C[手写数字识别模型]
    #B -->|其他物体| D[电子外设/工具识别模型]
    #C --> E[输出数字结果]
    #D --> F[输出物体类别]

import pyb, machine
import sensor, image, time, math
import os, tf
from machine import UART
from machine import Pin # 从pyb导入Pin


sensor.reset() #初始化感光元件
sensor.set_pixformat(sensor.RGB565) #设置为RGB565色彩空间
sensor.set_framesize(sensor.QVGA) #设置图像的大小
sensor.set_auto_exposure(False, exposure_us = 1100) #自动曝光关闭
sensor.set_brightness(700) # 设置图像亮度 越大越亮
sensor.set_auto_gain(True)  # must turn this off to prevent image washout...
sensor.set_auto_whitebal(True,(0,0x80,0))  # must turn this off to prevent image washout...
sensor.skip_frames(time = 1000) #等待1000ms使感光元件稳定



Light = pyb.LED(4) #定义补光灯


Light.on() #开启补光灯

clock = time.clock()

get_threshold = 0  # 调试灰度二值化阈值
debug = 1  #  1：调试模式，0：运行模式



if get_threshold == 1:
    while(True):
        clock.tick()
        img = sensor.snapshot()
        img.lens_corr(1.8) # 畸变校正
        img = img.to_grayscale()  #  .binary([hand_thresholds ]).invert()
        print(clock.fps(), "fps")



first_calss = 0 # 初步分类


uart = machine.UART(1,115200)   # 定义串口1，波特率115200

# 字典
scores_sum = {}
appear_sum = {}
average_scores = {}


# 目标分类
min_scores = 50 # 最小置信(100-->1)
max_count = 1 # 最大计数
max_count1 = 1 # 最大计数


# 目标对正

thresholds = [(68, 100, 20, -8, 3, -10)]      # 三分类块阈值


# 图片分类识别区域

classify_roi = (60,2,200,198)#xywh 黄框

target = {
       # 电子外设
       'mouse':101,                # 鼠标
       'keyboard':102,             # 键盘
       'display':103,              # 显示器
       'headphones':104,           # 头戴式耳机
       'audio':105,                # 音响
       'printer':106,              # 打印机
       'phone':107,                # 手机
       # 交通常用工具
       'wrench':108,               # 扳手
       'screwdriver':109,          # 螺丝刀
       'drill':110,               # 手电钻
       'pincers':111,             # 钳子
       'multimeter':112,          # 万用表
       'oscillograph':113,        # 示波器
       'iron':114,                # 电烙铁
       'tapemeasure':115,         # 卷尺

       'other':116

       }
target_display = {
       # 电子外设
       'mouse':"鼠标",                # 鼠标
       'keyboard':"键盘",             # 键盘
       'display':"显示器",              # 显示器
       'headphones':"头戴式耳机",           # 头戴式耳机
       'audio':"音响",                # 音响
       'printer':"打印机",              # 打印机
       'phone':"手机",                # 手机
       # 交通常用工具
       'wrench':"扳手",               # 扳手
       'screwdriver':"螺丝刀",          # 螺丝刀
       'drill':"手电钻",               # 手电钻
       'pincers':"钳子",             # 钳子
       'multimeter':"万用表",          # 万用表
       'oscillograph':"示波器",        # 示波器
       'iron':"电烙铁",                # 电烙铁
       'tapemeasure':"卷尺",         # 卷尺

       'other':116

       }


target_first = {
       'write':1,         # 1类
       'object':2,        # 2类
       'other':3}


net_path_first = None
net_first = None
labels_first = None
net_path = None
net = None
labels = None
hand_net_path = None
hand_net = None
hand_labels = None

while(True):

    print("模型加载中")

    try :
        # 加载初步模型
        #net_path_first = "big_V1.tflite"   # 定义模型的路径
        #net_path_first = "big_V49.tflite"   # 定义模型的路径
        net_path_first = "big_V418.tflite"   # 定义模型的路径
        print("%s模型加载成功!"%net_path_first)
        labels_first = [line.rstrip() for line in open("/sd/lables_big.txt")]   # 加载标签
        print("%s标签加载成功!"%labels_first)
        net_first = tf.load(net_path_first, load_to_fb=True)  # 加载模型
        print("%s模型装载加载成功!"%net_first)
        if net_first is None:
            continue  # 进入下一轮循环

        print("初步模型加成功！")
    except :
         print("初步模型加载失败！")
    try :
        ## 加载目标模型
        net_path = "object_V45.tflite"   # 定义模型的路径
        #net_path = "object_V49.tflite"   # 定义模型的路径
        labels = [line.rstrip() for line in open("/sd/lables_object.txt")]   # 加载标签
        net = tf.load(net_path, load_to_fb=True)  # 加载模型
        if net is None:
            continue  # 进入下一轮循环

        print("目标模型加成功！")
    except :
         print("目标模型加载失败！")
    try :
        hand_net_path = "write_num_V2.tflite"
        hand_labels = [line.rstrip() for line in open("/sd/labels_number.txt")]   # 加载标签 "labels_number.txt"
        hand_net = tf.load(hand_net_path , load_to_fb=True)
        if hand_net is None:
            continue  # 进入下一轮循环
        print("手写模型加成功！")
    except :
         print("手写模型加载失败！")
    if hand_net != None and net != None and net_first != None:
        break

############################################################################

# 图片旋转 1 0度 2 90度 3 4

# 图片旋转 1 0度 2 90度 3 4
def rotate(num, img):
    if num == 1:
        # 0 degrees: no changes
        return img.replace(vflip=False, hmirror=False, transpose=False)
    elif num == 2:
        # 90° clockwise: transpose + horizontal flip
        return img.replace(vflip=True,  hmirror=False, transpose=True)
    elif num == 3:
        # 180°: vertical + horizontal flip
        return img.replace(vflip=True, hmirror=True, transpose=False)
    elif num == 4:
        # 270° clockwise: transpose + vertical flip
        return img.replace(vflip=False, hmirror=True,  transpose=True)
    else:
        raise ValueError("Invalid rotation number. Use 1, 2, 3, or 4.")



def expand_roi(t,expand =1.4):

    center_x  = t[0] + t[2]/2
    center_y  = t[1] + t[3]/2
    # 计算中心坐标


    # 计算新尺寸
    new_w = expand * t[2] *1.15
    new_h = expand * t[3]

    # 计算新边界
    new_x = int(center_x - new_w / 2)
    new_y = int(center_y - new_h / 2)
    new_w = int(new_w)
    new_h = int(new_h)

    ## 裁剪到图像范围内
    #new_left = max(0.0, new_left)
    #new_right = min(float(320, new_right)
    #new_top = max(0.0, new_top)
    #new_bottom = min(float(240), new_bottom)

    ## 防止负尺寸
    #new_w = max(0.0, new_right - new_left)
    #new_h = max(0.0, new_bottom - new_top)

    return (new_x, new_y, new_w, new_h)


hand_roi = [30,30,260,180]
hand_roi  = (21,5,260,150)#xywh 黄框
hand_thresholds = (0, 121) # 白色是追踪值
#thresholds = (64, 85, -4, 55, -6, 23)
def find_digits_roi(img):
    img_roi = img.copy()
    num_roi = []
    for blob in img_roi.find_blobs([hand_thresholds ],roi= hand_roi, pixels_threshold=500, area_threshold=500, merge=True):
        rect = blob.rect()
        #img.draw_rectangle(rect, color=(255, 0, 0))
        rect =  expand_roi(rect)
        num_roi.append(rect)
        #img.draw_rectangle(rect, color=(255, 0, 255))

    return sorted(num_roi, key=lambda x: x[0])

## 合并区域
def find_connect_roi(img):
    img_roi = img.copy()
    # 用于保存所有区域的坐标
    regions = []
    for blob in img_roi.find_blobs([hand_thresholds], roi=hand_roi , pixels_threshold=500, area_threshold=500, merge=True):
        rect = blob.rect()
        rect = expand_roi(rect)  # 假设 expand_roi 返回 (x, y, w, h)
        regions.append(rect)

    # 如果没有检测到区域，直接返回空
    if not regions:
        return None

    # 合并所有区域为一个包围矩形
    min_x = min(r[0] for r in regions)
    min_y = min(r[1] for r in regions)
    max_x = max(r[0] + r[2] for r in regions)
    max_y = max(r[1] + r[3] for r in regions)
    merged_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

    return merged_rect



def shrink_roi(t,expand =0.8):

    center_x  = t[0] + t[2]/2
    center_y  = t[1] + t[3]/2
    # 计算中心坐标


    # 计算新尺寸
    new_w = expand * t[2]
    new_h = expand * t[3]

    # 计算新边界
    new_x = int(center_x - new_w / 2)
    new_y = int(center_y - new_h / 2)
    new_w = int(new_w)
    new_h = int(new_h)

    return (new_x, new_y, new_w, new_h)


roi_writhe = [30,0,260,160]
thresholds_writhe = (174, 255) # 白色是追踪值
def find_digits_writhe_roi(img):
    img_roi = img.copy()
    for blob in img_roi.find_blobs([thresholds_writhe],roi= roi_writhe, pixels_threshold=500, area_threshold=500, merge=True):
        roi_black = blob.rect()
        roi_black = shrink_roi(roi_black,0.7)

    return roi_black

roi_black = [30,0,260,180]
thresholds_black = (0, 167)  # 黑色是追踪值
#thresholds = (64, 85, -4, 55, -6, 23)
def find_digits_num_roi(img):
    img_roi = img.copy()
    roi_black = find_digits_writhe_roi(img_roi)  # 查找数字区域
    num_roi = []
    for blob in img_roi.find_blobs([thresholds_black],roi= roi_black, pixels_threshold=50, area_threshold=50, merge=True):
        rect = blob.rect()
        #print("black")
        #print(rect)
        #img.draw_rectangle(rect, color=(255, 0, 0))
        rect =  expand_roi(rect)
        num_roi.append(rect)
        #img.draw_rectangle(rect, color=(255, 0, 255))

    return sorted(num_roi, key=lambda x: x[0])

# merge_all_rects函数
def merge_all_rects(rectangles):
    if not rectangles:
        return (0, 0, 0, 0)
    new_x, new_y, new_w, new_h = rectangles[0]
    for rect in rectangles[1:2]:
        x, y, w, h = rect
        new_x = min(new_x, x)
        new_y = min(new_y, y)
        right = x + w
        bottom = y + h
        new_w = max(new_w, right - new_x)
        new_h = max(new_h, bottom - new_y)
    return (new_x, new_y, new_w, new_h)





def capture_and_preprocess_image(img2,rnum):

    """图像采集与基础预处理"""


    img_gray = img2.lens_corr(1.8).to_grayscale()  # 转换为灰度图

    rects = find_digits_num_roi(img_gray)  # 查找数字区域

    # 合并所有区域
    merge_all_rect = merge_all_rects(rects)

    #print(merge_all_rect)


    img1 = img_gray.copy(roi=merge_all_rect) #裁剪出目标区域的图像。
    img1 = rotate(rnum,img1)
    img = img1.copy()

    rects1 = find_digits_num_roi(img)  # 查找数字区域

    #print("区域")
    #print(rects1)


    img = img_gray.binary([thresholds_black]).invert()

    img.erode(3)
    #img.dilate(2)  # 再膨胀恢复数字形状

    return img,rects1


#  缩放图片
def crop_and_scale_region(region,r, target_size):
    """裁剪并缩放区域"""
    cropped = region.copy(roi=r)
    scale_x = target_size[0] / cropped.width()
    scale_y = target_size[1] / cropped.height()
    return cropped.scale(scale_x, scale_y)




def RGNN_Find(hand_net = None,hand_labels = None,rnum = 1):

    img = sensor.snapshot()
    img,rect = capture_and_preprocess_image(img,rnum)


    num = []  # 初始化数字列表
    score = []  # 初始化数字列表
    for n, r in enumerate(rect[:5]):  # 只处理前5个矩形,防止内存溢出，最大只能识别5位数
        if n >= 2:
            break  # 这个break在这里不是必需的，但它强调了循环的限制
        img.draw_rectangle(r, color=(255, 0, 0))  # 绘制矩形框

        cropped_img = crop_and_scale_region(img,r,[32,32])

        #for obj in tf.classify(net, img, r, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        for obj in tf.classify(hand_net,cropped_img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
            sorted_list = sorted(zip(hand_labels, obj.output()), key=lambda x: x[1], reverse=True)
            recognized_num = sorted_list[0][0]
            num.append(recognized_num)  # 使用append添加元素
            recognized_score = sorted_list[0][1]
            score.append(recognized_score)
            #print(sorted_list)
            #print(f"Recognized number: {recognized_num}")
            img.draw_string(r[0], r[1], f"{recognized_num}", color=(255, 0, 0), scale=2)

    try:
        # 将列表中的每个数字转换成字符串，并连接它们
        num_str = ''.join(map(str, num))
        # 将连接后的字符串转换成整数
        num_int = int(num_str)

        del img,num,num_str
        gc.collect()  # 显式触发垃圾收集，尝试回收内存
        return num_int,sum(score)
    except :
        print("数字拼接失败！")

        return -1,-1


def hand_identify():

    # 主循环函数，完成手写数字识别和结果输出

    # 定义角度与旋转标识的映射关系
    rotations = [
        (4, "270"),    # 旋转4次对应270度
        (3, "180"),
        (2, "90"),
        (1, "0")
    ]

    results = {}  # 存储各角度识别结果

    for rnum, angle_desc in rotations:
        try:
            num, score = RGNN_Find(hand_net, hand_labels, rnum)
            if num != -1 and score > 0:
                results[angle_desc] = (num, score)
                #print(f"Recognized {angle_desc} numbers: {num}, Score: {score:.2f}")
            else:
                print(f"No valid digits recognized at {angle_desc} degrees")
        except Exception as e:
            print(f"Error processing {angle_desc} degree image: {e}")

    # 选择最优结果
    if not results:
        print("No valid recognition results")
        return -1  # 无有效识别结果

    max_score = max(score for _, score in results.values())
    best_results = [res for res in results.values() if res[1] == max_score]

    # 处理平局情况（取数值最大的数字）
    best_num = max(res[0] for res in best_results)
    best_angle = next(angle for angle, (_, score) in results.items() if score == max_score)

    print(f"\nBest result: {best_num} ({best_angle} degrees), Score: {max_score:.2f}")

    # 串口输出优化
    output_packet = bytearray([0x1B, best_num, 0xFF])
    try:
        uart.write(output_packet)
        print("手写数字识别：发送完毕！")
    except OSError as e:
        print(f"串口写入失败: {e}")

    # 清理资源
    gc.collect()

    return 1  # 执行成功,回归串口

    #################################################################################


# 将接收的数据由bytes类型转化为str类型
def byteToStr(data):
    return ''.join(["%02x"%x for x in data]).strip()



# 初步分类 #######################################################
max_name_first = None
def identify_first(labels = None,net = None):
    img = sensor.snapshot()
    cropped_img = img.copy(roi=classify_roi) # 复制识别区域
    img.draw_rectangle(classify_roi,color=(255,255,0),thickness=1)
    max_name_first = 999
    print("大类正在识别")
    for obj in tf.classify(net,cropped_img,min_scale=1.0, scale_mul=0.5, x_overlap=0.0, y_overlap=0.0):
        #print("**********\nTop 1 Detections at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
#        print(sorted_list)
        # 打印准确率最高的结果
        # print("obj")
        first = sorted_list[0] #准确率最高
        name,scores = first
        if scores*100 >= min_scores:
            if name not in scores_sum:
                scores_sum[name] = 0
                appear_sum[name] = 0

            scores_sum[name] += scores*100
            appear_sum[name] += 1

            if max(appear_sum.values()) >= max_count1:
                for key,val in appear_sum.items():
                    if val == max_count1:
                        average_scores[key] = scores_sum[key] / appear_sum[key]

                max_name_first = max(average_scores, key=average_scores.get)

                scores_sum.clear()
                appear_sum.clear()
                average_scores.clear()
    print("大类识别结果是： %s"%max_name_first)
    num = target_first[max_name_first]
    #print(num)

    del cropped_img,first,img
    try :
        scores_sum.clear()
        appear_sum.clear()
        average_scores.clear()
        return num
    except :
        print("目标数据出错")

        return 3




# 细分类 #######################################################

def identify(labels = None,net = None):
    #global first_calss #声明我们在函数内部使用的是在函数外部定义的全局变量a
    max_name = -1
    print("小类正在识别")

    for  count1 in range(3) :  # 物体识别3次，取最后一次（应该取众数）
        img = sensor.snapshot()
        #cropped_img = img.copy(roi=classify_roi) # 复制识别区域
        cropped_img = crop_and_scale_region(img,classify_roi, [192,192])
        #print(cropped_img)
        img.draw_rectangle(classify_roi,color=(255,255,0),thickness=2)
        #cropped_img = cropped_img.resize(128,128)
        for obj in tf.classify(net,cropped_img,min_scale=1.0, scale_mul=0.5, x_overlap=0.0, y_overlap=0.0):
            #print("**********\nTop 1 Detections at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
            sorted_list = sorted(zip(labels, obj.output()), key = lambda x: x[1], reverse = True)
            #print(sorted_list)
            # 打印准确率最高的结果
            #print("obj")
            first = sorted_list[0] #准确率最高
            name,scores = first
            if scores*100 >= min_scores:
                if name not in scores_sum:
                    scores_sum[name] = 0
                    appear_sum[name] = 0

                scores_sum[name] += scores*100
                appear_sum[name] += 1

                if max(appear_sum.values()) >= max_count:
                    for key,val in appear_sum.items():
                        if val == max_count:
                            average_scores[key] = scores_sum[key] / appear_sum[key]

                    max_name = max(average_scores, key=average_scores.get)

                    print(max_name)

                    scores_sum.clear()
                    appear_sum.clear()
                    average_scores.clear()
                    #all_blob.clear()
    try:
        print("小类识别结果是： %s"%target_display[max_name])
    except :
        print("中文输出错误！")
    try :
        uart.write(bytes([0x1B, target[max_name], 0xFF]))
        print("物体识别：发送完毕！")
    except :
         print("物体识别：发送失败！")


    #first_calss = 0 #下一次识别启用初步分类

    del max_name
    return 1 # 启用串口接收


# 图片检测，返回中心点,默认无圆环
locate_roi = (20,1,300,160)      # 对正识别范围(X)
thresholds_pic = (68, 100)     # 查找阈值  建议使用 除蓝色和白色背景以外的阈值查找

def pic_center():
    img = sensor.snapshot()  # 释放转灰度和校正滤波
    num_roi = []
    for blob in img.find_blobs([thresholds_pic],roi= locate_roi, pixels_threshold=500, area_threshold=500, merge=True):
        rect = blob.rect()

    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    cx = x+int(w/2)
    cy = y+int(h/2)


    if cx < 30 :
        print("out range !")
        # 此处开补光灯灯
        cx = 30
    if cy < 30 :
        print("out range !")
        # 此处开补光灯灯
        cy = 30

    elif cx > 250 or cy > 250 :
        print("out range !")
        # 此处开补光灯灯
#            cx = 160
#            cy = 139


    else :
        try :
            uart.write(bytes([0x1A, cx, cy, 0xFF]))
            print("位置发送成功")
            print(cx,cy)
        except :
             print("位置发送失败！")



                # 此处关补光灯灯
    img.draw_cross(cx, cy,color=(255,0,0))
    img.draw_rectangle(locate_roi,color=(255,255,255),thickness=2)





uart_flag = 1 # 启用串口接收
flag = 1
if debug == 1:
    while(True):
        clock.tick()
        #openart接收数据
        if uart_flag == 1:
            print("启用串口\t\t")
            uart_read_data=0

            data_len=uart.any()

            if(data_len):
                read_data = uart.read(1) # 读取串口数据
                uart_read_data=byteToStr(read_data)
                #print(uart_read_data)

            uart_read_data = 'fa'
            if uart_read_data == 'fa': #如果是目标
                uart_flag = 0
            elif uart_read_data == 'fb': #如果是对正
                pic_center() # 位置查找
            del data_len ,uart_read_data
        else :
                first_calss = 2
                if first_calss == 0:
                    first_calss = identify_first(labels_first,net_first) # 初步分类
                    #first_calss = 2 # 初步分类 强制物体识别
                    #if flag == 1 :
                        #first_calss = 1 # 初步分类
                    #else :
                        #first_calss = 2 # 初步分类
                elif first_calss == 1 :
                    uart_flag = hand_identify()  # 大1类识别  数字识别
                    flag = 2
                    first_calss = 0 # 下一次继续大类识别
                elif first_calss == 2 :
                    uart_flag = identify(labels,net) # 大2类识别  物体识别
                    flag = 1
                    first_calss = 0 # 下一次继续大类识别
                    #gc.collect()  # 显式触发垃圾收集，尝试回收内存
                else :
                    print('类别是：%s'%first_calss)
                    print('识别出错')
                    first_calss = 0 # 下一次继续大类识别

        #first_calss = 0 # 下一次继续大类识别

        gc.collect()  # 显式触发垃圾收集，尝试回收内存
        #print(clock.fps(), "fps")
        #print(gc.mem_free())    #显示剩余内存 largest_blob


while(True):
    try :
        clock.tick()
        #openart接收数据
        if uart_flag == 1:
            print("启用串口\t\t")
            uart_read_data=0

            data_len=uart.any()

            if(data_len):
                read_data = uart.read(1) # 读取串口数据
                uart_read_data=byteToStr(read_data)
                #print(uart_read_data)

            uart_read_data = 'fa'
            if uart_read_data == 'fa': #如果是目标
                uart_flag = 0
            elif uart_read_data == 'fb': #如果是对正
                pic_center() # 位置查找
            del data_len ,uart_read_data
        else :
                #first_calss = 2
                if first_calss == 0:
                    first_calss = identify_first(labels_first,net_first) # 初步分类
                    print(first_calss)
                    #first_calss = 2
                    #first_calss = 1 # 初步分类
                    #if flag == 1 :
                        #first_calss = 1 # 初步分类
                    #else :
                        #first_calss = 2 # 初步分类

                else :
                    if first_calss == 1 :
                        print("数字识别")
                        uart_flag = hand_identify()  # 大1类识别  数字识别
                        flag = 2
                    elif first_calss == 2 :
                        print("物体识别")
                        uart_flag = identify(labels,net) # 大2类识别  物体识别
                        flag = 1
                        #gc.collect()  # 显式触发垃圾收集，尝试回收内存
                    else :
                        print('类别是：%s'%first_calss)
                        print('识别出错')
                    first_calss = 0 # 下一次继续大类识别
        #first_calss = 0 # 下一次继续大类识别

        gc.collect()  # 显式触发垃圾收集，尝试回收内存
        print(clock.fps(), "fps")
        print(gc.mem_free())    #显示剩余内存 largest_blob
    except :
         print(f"运行失败！")
         if debug == 0 :
            continue  # 进入下一轮循环
