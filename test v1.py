import cv2 as cv
import gbvision as gbv
import numpy as np


pic = cv.imread('road.jpg')
_mask = cv.imread('mask.jpg')
win = gbv.FeedWindow('original')
win.open()

# ______________    init    ___________________________________________________


bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
subj = pic.copy()


mask = np.zeros(_mask.shape[:2], np.uint8)


rect = cv.boundingRect(gbv.find_contours(_mask))

mask, bgdModel, fgdModel = cv.grabCut(subj, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = subj * mask[:, :, np.newaxis]


cv.rectangle(pic, rect[0], rect[1], (255, 0, 0), 4)
while 1:
    win.show_frame(pic)


cv.destroyAllWindows()


