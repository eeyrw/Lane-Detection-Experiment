import cv2
import numpy as np

# 读取名称为 p19.jpg的图片
img = cv2.imread(r"E:\CULane\driver_193_90frame\06042016_0513.MP4\01890.jpg",1)
img_org = cv2.imread(r"E:\CULane\driver_193_90frame\06042016_0513.MP4\01890.jpg",1)

# 得到图片的高和宽
img_height,img_width = img.shape[:2]

# 定义对应的点
points1 = np.float32([[553,424], [839,424], [0,583], [1233,583]])
points2 = np.float32([[519,400], [977,400], [519,900], [977,900]])

# 计算得到转换矩阵
M = cv2.getPerspectiveTransform(points1, points2)
print(M)

# 实现透视变换转换
processed = cv2.warpPerspective(img,M,(img_width, img_height+600))

# 显示原图和处理后的图像
cv2.imshow("org",img_org)
cv2.imshow("processed",processed)

cv2.waitKey(0)