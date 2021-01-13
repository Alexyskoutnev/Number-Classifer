import urllib.request
import cv2
import numpy as np
import time
from scipy import ndimage
import math

def read(Network):
    camera = cv2.VideoCapture('http://10.0.0.198:8080/video')
    Lower_Black = np.array([0, 0, 0])
    Upper_Black = np.array([100, 100, 100])
    center = None
    while True:
        (_, img) = camera.read()
        #img = cv2.imread('Sample_Number_3.jpg', 1)
        #img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        # mask_inrange = cv2.inRange(hsv, Lower_Black, Upper_Black)
        # #mask_morph_open = cv2.morphologyEx(mask_inrange, cv2.MORPH_OPEN, kernel)
        # #mask_dilate = cv2.dilate(mask_inrange, kernel, iterations=2)
        # #mask = mask_dilate
        # res = cv2.bitwise_and(img, img, mask=mask_inrange)

        # #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("inrange", mask_inrange)
        # #cv2.imshow("erode", mask_erode)
#        cv2.imshow("morph open", mask_morph_open)
        #cv2.imshow("morph closed", mask_morph_closed)
        # cv2.imshow("mask dilate", mask_dilate)
        # cv2.imshow("res", res)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        get_calculated_rectangles = get_rectangles(contours, hierarchy)
        cv2.imshow('thresh', thresh)
        try:
            for Rectangle in get_calculated_rectangles:
                filename = 'EXAMPLE_OUTPUT.jpg'
                x, y, h, w = Rectangle
                digit_image = img[y:y + w, x:x + h]
                img_processed = process(digit_image)
                # gray_digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
                # gray_digit_image = cv2.resize(gray_digit_image, (28, 28))
                #digit_data = np.array(img_processed)
                # digit_data = process(digit_image)
                # cv2.imshow("porcess", digit_data)
                digit_data = (img_processed.flatten())/255.0
                digit_data_new = digit_data.reshape(784, 1)
                #cv2.imwrite(filename, gray_digit_image)
                #cv2.imshow("Final Image111", gray_digit_image)
                output_digit = np.argmax(Network.feedforward(digit_data_new))

                cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
                cv2.putText(img, str(output_digit), (x + h, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except:
            print("broke")
            pass
        #Show Image
        resized_image = cv2.resize(img, (1000, 1000))
        cv2.imshow("Digits Recognition Real Time", resized_image)

        #Break Image
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #Turn off Camera
    camera.release()
    #cv2.waitKey()
    cv2.destroyWindow()

def get_rectangles(cont, hier):
    try:
        hier = hier[0]
    except:
        return None
    rectangles = [cv2.boundingRect(contours) for contours in cont]
    digit_rectangles = []
    u, indices = np.unique(hier[:, -1], return_inverse=True)
    common_heirarchy = u[np.argmax(np.bincount(indices))]
    for rec, hr in zip(rectangles, hier):
        x,y,w,h = rec
        if ((w*h) > 500) and (30 <= w <= 200) and (30 <= h <= 200) and (hr[3] == common_heirarchy):
            digit_rectangles.append(rec)
    return digit_rectangles


def process(img):
    gray_digit_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, gray_digit_image = cv2.threshold(gray_digit_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray_digit_image = cv2.resize(255 - gray_digit_image, (28, 28))
    rows, cols = gray_digit_image.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray_digit_image = cv2.resize(gray_digit_image, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray_digit_image = cv2.resize(gray_digit_image, (cols, rows))

    #Add the missing corner pixels
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray_digit_image = np.lib.pad(gray_digit_image, (rowsPadding, colsPadding), 'constant')

    shiftx, shifty = getBestShift(gray_digit_image)
    shifted = shift(gray_digit_image, shiftx, shifty)
    gray_digit_image = shifted

    return gray_digit_image

def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

def getBestShift(img):
    M = cv2.moments(img)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty


