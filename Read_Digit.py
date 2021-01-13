import urllib.request
import cv2
import numpy as np
import time
from scipy import ndimage

def read(Network):
    array_digits_data = []
    camera = cv2.VideoCapture('http://10.0.0.198:8080/video')
    Lower_Black = np.array([0, 0, 0])
    Upper_Black = np.array([100, 100, 100])
    center = None
    while True:
        # img = cv2.imread('Multiple_Numbers.jpg',1)
        (_, img) = camera.read()
        #img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
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
                gray_digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
                gray_digit_image = cv2.resize(gray_digit_image, (28, 28))
                digit_data = np.array(gray_digit_image)
                # digit_data = process(digit_image)
                # cv2.imshow("porcess", digit_data)
                digit_data = digit_data.astype('float32').flatten() / 255.0
                digit_data_new = digit_data.reshape(784, 1)
                #cv2.imwrite(filename, gray_digit_image)
                array_digits_data.append(digit_data_new)
                #cv2.imshow("Final Image111", gray_digit_image)
                output_digit = np.argmax(Network.feedforward(digit_data_new))
                cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
                cv2.putText(img, str(output_digit), (x + h, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except:
            pass
        #Show Image
        resized_image = cv2.resize(img, (800, 800))
        cv2.imshow("Digits Recognition Real Time", resized_image)

        #Break Image
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #Turn off Camera
    camera.release()
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
        if ((w*h) > 300) and (20 <= w <= 200) and (20 <= h <= 200) and (hr[3] == common_heirarchy):
            digit_rectangles.append(rec)
    return digit_rectangles

# def process(img):
#     digit_image = -np.ones(img.shape)
#
#     height, width = img.shape
#
#     predSet_ret = []
#
#     """
#     crop into several images
#     """
#     for cropped_width in range(100, 300, 20):
#         for cropped_height in range(100, 300, 20):
#             for shift_x in range(0, width - cropped_width, int(cropped_width / 4)):
#                 for shift_y in range(0, height - cropped_height, int(cropped_height / 4)):
#                     gray = img[shift_y:shift_y + cropped_height, shift_x:shift_x + cropped_width]
#                     if np.count_nonzero(gray) <= 20:
#                         continue
#
#                     if (np.sum(gray[0]) != 0) or (np.sum(gray[:, 0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,
#                                                                                                                  -1]) != 0):
#                         continue
#
#                     top_left = np.array([shift_y, shift_x])
#                     bottom_right = np.array([shift_y + cropped_height, shift_x + cropped_width])
#
#                     while np.sum(gray[0]) == 0:
#                         top_left[0] += 1
#                         gray = gray[1:]
#
#                     while np.sum(gray[:, 0]) == 0:
#                         top_left[1] += 1
#                         gray = np.delete(gray, 0, 1)
#
#                     while np.sum(gray[-1]) == 0:
#                         bottom_right[0] -= 1
#                         gray = gray[:-1]
#
#                     while np.sum(gray[:, -1]) == 0:
#                         bottom_right[1] -= 1
#                         gray = np.delete(gray, -1, 1)
#
#                     actual_w_h = bottom_right - top_left
#                     if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] + 1) >
#                             0.2 * actual_w_h[0] * actual_w_h[1]):
#                         continue
#
#                     rows, cols = gray.shape
#                     compl_dif = abs(rows - cols)
#                     half_Sm = int(compl_dif / 2)
#                     half_Big = half_Sm if half_Sm * 2 == compl_dif else half_Sm + 1
#                     if rows > cols:
#                         gray = np.lib.pad(gray, ((0, 0), (half_Sm, half_Big)), 'constant')
#                     else:
#                         gray = np.lib.pad(gray, ((half_Sm, half_Big), (0, 0)), 'constant')
#
#                     gray = cv2.resize(gray, (20, 20))
#                     gray = np.lib.pad(gray, ((4, 4), (4, 4)), 'constant')
#
#                     shiftx, shifty = getBestShift(gray)
#                     shifted = shift(gray, shiftx, shifty)
#                     gray = shifted
#     return gray
#
# def shift(img, sx, sy):
#     rows, cols = img.shape
#     M = np.float32([[1, 0, sx], [0, 1, sy]])
#     shifted = cv2.warpAffine(img, M, (cols, rows))
#     return shifted
#
# def getBestShift(img):
#     cy,cx = ndimage.measurements.center_of_mass(img)
#     print(cy,cx)
#
#     rows,cols = img.shape
#     shiftx = np.round(cols/2.0-cx).astype(int)
#     shifty = np.round(rows/2.0-cy).astype(int)
#     return shiftx, shifty


