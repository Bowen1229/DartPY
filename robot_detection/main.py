import sys
import threading
import numpy as np
import cv2
from MvCameraControl_class import *

# 参数设置
history_size = 5  # 历史帧记录数量
brightness_factor = 0.8  # 亮度因子Q
static_threshold = 1.0  # 判断目标是否静Q止的距离阈值（像素）
recent_weight_factor = 1.5  # 最近帧的权重因子

# 历史目标记录
target_positions_history = []
current_target = None  # 当前真实目标

# 定义全局退出标志
g_bExit = False


# 获取图像并处理
def work_thread2(cam, pData, nDataSize):
    global g_bExit
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    img_buff = None
    cv2.namedWindow('test')

    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # OpenCV 要用 BGR，不能使用 RGB
            nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
            if img_buff is None:
                img_buff = (c_ubyte * stFrameInfo.nFrameLen)()

            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                sys.exit()
            else:  # 显示及后处理
                img_buff = np.frombuffer(stConvertParam.pDstBuffer, dtype=np.uint8)
                img_buff = img_buff.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, 3)

                # 处理图像
                mask = preprocess_image(img_buff)
                process_contours(img_buff, mask)
                cv2.imshow('Processed Video', img_buff)  # 显示处理后的图像

        if g_bExit:
            break


# 图像预处理函数
def preprocess_image(input_image):
    darkened_image = input_image.astype(np.float32)
    darkened_image *= brightness_factor
    darkened_image = darkened_image.clip(0, 255).astype(np.uint8)

    hsv_image = cv2.cvtColor(darkened_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


# 轮廓处理和目标检测
def process_contours(input_image, mask):
    global target_positions_history
    global current_target

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_targets = []

    for contour in contours:
        if len(contour) >= 5:
            fitted_ellipse = cv2.fitEllipse(contour)
            ellipse_center = fitted_ellipse[0]
            current_targets.append(ellipse_center)
            cv2.ellipse(input_image, fitted_ellipse, (255, 0, 0), 2)

    target_positions_history.append(current_targets)
    if len(target_positions_history) > history_size:
        target_positions_history.pop(0)

    static_candidates = []
    if len(target_positions_history) >= history_size:
        for target in current_targets:
            position_history = []
            for history in target_positions_history:
                for old_target in history:
                    dist = np.sqrt((target[0] - old_target[0]) ** 2 + (target[1] - old_target[1]) ** 2)
                    if dist < static_threshold:
                        position_history.append(old_target)
                        break

            if len(position_history) >= 3 and calculate_movement_stability(position_history, static_threshold):
                static_candidates.append(target)

    if static_candidates:
        current_target = min(static_candidates, key=lambda t: np.sqrt(
            (t[0] - current_target[0]) ** 2 + (t[1] - current_target[1]) ** 2)) if current_target else \
        static_candidates[0]
    elif current_targets:
        current_target = min(current_targets, key=lambda t: np.sqrt(
            (t[0] - current_target[0]) ** 2 + (t[1] - current_target[1]) ** 2)) if current_target else current_targets[0]

    if current_target:
        cv2.circle(input_image, (int(current_target[0]), int(current_target[1])), 5, (0, 255, 0), -1)
        cv2.putText(input_image, f"Target: ({int(current_target[0])}, {int(current_target[1])})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    height, width, _ = input_image.shape
    cv2.line(input_image, (width // 2, 0), (width // 2, height), (0, 255, 255), 1)
    cv2.line(input_image, (0, height // 2), (width, height // 2), (0, 255, 255), 1)


# 检测目标的稳定性
def calculate_movement_stability(position_history, threshold):
    movement_sum = 0
    for i in range(1, len(position_history)):
        dx = position_history[i][0] - position_history[i - 1][0]
        dy = position_history[i][1] - position_history[i - 1][1]
        movement_sum += np.sqrt(dx ** 2 + dy ** 2)
    return movement_sum / len(position_history) < threshold


# 主程序
def main():
    # 获取设备信息并连接相机
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)

    # 选择第一个设备连接
    nConnectionNum = 0
    if nConnectionNum >= deviceList.nDeviceNum:
        print("input error!")
        sys.exit()

    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[nConnectionNum], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # 开启处理图像的线程
    pData = (c_ubyte * 1024 * 1024)()
    nDataSize = 1024 * 1024
    work_thread = threading.Thread(target=work_thread2, args=(cam, pData, nDataSize))
    work_thread.start()

    # 等待线程结束
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            global g_bExit
            g_bExit = True
            break

    # 关闭设备和清理
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
