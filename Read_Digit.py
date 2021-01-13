import urllib.request
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
#cap = cv2.VideoCapture('http://192.168.18.37:8090/video')

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
        kernel = np.ones((5, 5), np.uint8)
        _, thresh = cv2.threshold(gray, 125, 255, 0)
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

        contours, heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        get_calculated_rectangles = get_rectangles(contours, heir)
        cv2.imshow('thresh', thresh)
        cv2.imshow('thresh', gray)

        for Rectangle in get_calculated_rectangles:
            filename = 'EXAMPLE_OUTPUT.jpg'
            x, y, h, w = Rectangle
            digit_image = img[y:y + w, x:x + h]
            gray_digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
            gray_digit_image = cv2.resize(gray_digit_image, (28, 28))
            digit_data = np.array(gray_digit_image)
            digit_data = digit_data.astype('float32').flatten() / 255.0
            digit_data_new = digit_data.reshape(784, 1)
            cv2.imwrite(filename, gray_digit_image)
            array_digits_data.append(digit_data_new)
            cv2.imshow("Final Image111", gray_digit_image)
            output_digit = np.argmax(Network.feedforward(digit_data_new))
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
            cv2.putText(img, str(output_digit), (x + h, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


            # for digit in contours:
            #     num = 0
            #     filename = 'EXAMPLE_OUTPUT' + str(num) + '.jpg'
            #     x, y, h, w = cv2.boundingRect(np.float32(digit))
            #     digit_image = img[y:y + w, x:x + h]
            #     gray_digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
            #     gray_digit_image = cv2.resize(gray_digit_image, (28, 28))
            #     digit_data = np.array(gray_digit_image)
            #     digit_data = digit_data.astype('float32').flatten()/ 255.0
            #     digit_data_new = digit_data.reshape(784,1)
            #     cv2.imwrite(filename, gray_digit_image)
            #     array_digits_data.append(digit_data_new)
            #     cv2.imshow("Final Image111", gray_digit_image)
            #     num += 1
            #     output_digit = np.argmax(Network.feedforward(digit_data_new))
            #     # print(Network.feedforward(digit_data_new), "output")
            #     # print(np.argmax(Network.feedforward(digit_data_new)), "Output")
            #     cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
            #     cv2.putText(img, str(output_digit), (x + h, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

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
    hierarchy = hier[0]
    rectangles = [cv2.boundingRect(contours) for contours in cont]
    digit_rectangles = []
    u, indices = np.unique(hierarchy[:, -1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    for rec,hr in zip(rectangles, hierarchy):
        x,y,w,h = rec
        if ((w*h) > 250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy:
            digit_rectangles.append(rec)
    return digit_rectangles



