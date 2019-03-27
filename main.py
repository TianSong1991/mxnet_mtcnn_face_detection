# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

original_path = '/******/Kevin_ubuntu/Object_detection/mxnet_mtcnn_face_detection/images_original'

landmark_path = '/******/Kevin_ubuntu/Object_detection/mxnet_mtcnn_face_detection/images_landmark'

datalist = []

def get_all_files(path1):
    files = os.listdir(path1)
    for file in files:
        file_path = os.path.join(path1,file)            
        if os.path.isdir(file_path):
            get_all_files(file_path)                  
        else:
            (name1,extension1) = os.path.splitext(file_path)
            if extension1 == '.jpg':
                datalist.append(file_path)
get_all_files(original_path)

for i in range(len(datalist)):

    img = cv2.imread(datalist[i])
    (name1,extension1) = os.path.splitext(datalist[i])
    (name2,extension2) = os.path.split(name1)

    # run detector
    results = detector.detect_face(img)

    if results is not None:

        total_boxes = results[0]#返回人脸概率与点坐标
        #print(total_boxes.shape[0])#打印输出图片中包含人脸的个数
        
        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255),5)
        cv2.imwrite(os.path.join(landmark_path,extension2+'_draw_'+str(i)+'.png'), draw)



#########显示图片中每一个脸并裁剪成小图片###############
# extract aligned face chips
    # points = results[1]
    # chips = detector.extract_image_chips(img, points, 144, 0.37)
    # for i, chip in enumerate(chips):
    #     cv2.imshow('chip_'+str(i), chip)
    #     cv2.imwrite('chip_'+str(i)+'.png', chip)
###################################################


#########在脸上绘制5个点################################################
    # for p in points:
    #     for i in range(5):
    #         cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

    # cv2.imshow("detection result", draw)
    # cv2.waitKey(0)
######################################################################  

# --------------
# test on camera
# --------------
'''
camera = cv2.VideoCapture(0)
while True:
    grab, frame = camera.read()
    img = cv2.resize(frame, (320,180))

    t1 = time.time()
    results = detector.detect_face(img)
    print 'time: ',time.time() - t1

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    cv2.imshow("detection result", draw)
    cv2.waitKey(30)
'''
