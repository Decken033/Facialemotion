import cv2
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import time
import os
import psutil
import threading

tf.compat.v1.disable_eager_execution()

# 全局变量
MODEL = None
CLASSIFIER = None
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
CPU_USAGE = []
MONITOR_CPU = True


def monitor_cpu_usage():
    """监控CPU使用率的线程函数"""
    global CPU_USAGE, MONITOR_CPU
    while MONITOR_CPU:
        CPU_USAGE.append(psutil.cpu_percent(interval=0.1))
        time.sleep(0.1)  # 每0.1秒采样一次


def load_models():
    """预加载模型，只在第一次调用时加载"""
    global MODEL, CLASSIFIER

    print("正在加载模型...")
    start_time = time.time()

    # 加载情绪识别模型
    json_file = open('Saved-Models/cnn_20250411_022421_acc_0.8351_structure.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    MODEL = model_from_json(loaded_model_json)
    MODEL.load_weights('Saved-Models/cnn_20250411_022421_acc_0.8351_weights.h5')

    # 编译模型，避免第一次预测时的编译延迟
    MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 预热模型，避免第一次推理的延迟
    dummy_input = np.zeros((1, 48, 48, 1))
    MODEL.predict(dummy_input)

    # 加载人脸检测器
    CLASSIFIER = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    # 预热人脸检测器
    dummy_blob = cv2.dnn.blobFromImage(np.zeros((300, 300, 3), dtype=np.uint8),
                                       1.0, (300, 300), (104.0, 177.0, 123.0))
    CLASSIFIER.setInput(dummy_blob)
    CLASSIFIER.forward()

    end_time = time.time()
    print(f"模型加载完成！耗时: {(end_time - start_time) * 1000:.2f}ms")


def predict_emotion(image_path):
    global MODEL, CLASSIFIER, EMOTIONS, CPU_USAGE, MONITOR_CPU

    # 确保模型已加载
    if MODEL is None or CLASSIFIER is None:
        load_models()

    # 启动CPU监控线程
    CPU_USAGE = []
    MONITOR_CPU = True
    cpu_thread = threading.Thread(target=monitor_cpu_usage)
    cpu_thread.daemon = True
    cpu_thread.start()

    start_time = time.time()

    # 读取和预处理图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        MONITOR_CPU = False
        return

    img = cv2.resize(img, (640, 480))  # 预缩放以加速处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建blob并进行人脸检测
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    CLASSIFIER.setInput(blob)
    detections = CLASSIFIER.forward()

    # 处理检测结果
    faces_detected = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (x, y, x2, y2) = box.astype("int")
            w, h = x2 - x, y2 - y
            faces_detected.append((x, y, w, h))

    # 处理检测到的人脸
    if len(faces_detected) == 0:
        print("未检测到人脸！")
    else:
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 确保坐标有效
            y = max(0, y)
            x = max(0, x)
            h = min(h, img.shape[0] - y)
            w = min(w, img.shape[1] - x)

            if h <= 0 or w <= 0:
                continue

            # 提取人脸ROI并进行预处理
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            # 预测情绪
            predictions = MODEL.predict(img_pixels, verbose=0)  # 添加verbose=0以减少输出
            max_index = int(np.argmax(predictions))
            predicted_emotion = EMOTIONS[max_index]

            # 在图像上绘制情绪标签
            cv2.putText(img, predicted_emotion, (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            print(f"检测到人脸，位置: (x={x}, y={y}, w={w}, h={h})，预测情绪: {predicted_emotion}")

        # 保存处理后的图像
        output_path = "output_with_emotion.jpg"
        cv2.imwrite(output_path, img)
        print(f"处理后的图像已保存为: {output_path}")

    # 停止CPU监控
    MONITOR_CPU = False
    cpu_thread.join(timeout=1)

    # 计算并显示延迟
    end_time = time.time()
    process_time = (end_time - start_time) * 1000
    print(f"端到端延迟: {process_time:.2f}ms")

    # 分析CPU使用率
    if CPU_USAGE:
        avg_cpu = sum(CPU_USAGE) / len(CPU_USAGE)
        max_cpu = max(CPU_USAGE)
        print(f"平均CPU占用率: {avg_cpu:.2f}%")
        print(f"最高CPU占用率: {max_cpu:.2f}%")

        if max_cpu <= 60:
            print("CPU占用率良好：≤60%")
        else:
            print("CPU占用率过高：>60%，考虑进一步优化或增加硬件资源")


def main():
    # 预加载模型，这样只需要加载一次
    load_models()

    # 检查是否安装了psutil
    try:
        import psutil
    except ImportError:
        print("警告: 未检测到psutil库，无法监控CPU使用率。请使用pip install psutil安装。")
        return

    # 解析命令行参数
    ap = argparse.ArgumentParser()
    ap.add_argument('image', help='path to input image file')
    args = vars(ap.parse_args())

    # 处理图像
    predict_emotion(args['image'])


if __name__ == "__main__":
    main()