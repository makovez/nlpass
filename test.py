import cv2

img = 'data/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg'

data = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
