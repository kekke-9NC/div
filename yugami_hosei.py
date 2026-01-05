import cv2
import numpy as np
import matplotlib.pyplot as plt

# マップの読み込み
map_x = np.load(r"C:\Users\kekke\Downloads\star_trail_calibration_output\distortion_map_x.npy")
map_y = np.load(r"C:\Users\kekke\Downloads\star_trail_calibration_output\distortion_map_y.npy")
img = cv2.imread(r"C:\Users\kekke\Downloads\test.jpg")

# マップを使って画像をリマッピング（補正）
undistorted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# 表示
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)), plt.title('Undistorted')
plt.show()