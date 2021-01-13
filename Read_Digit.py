import urllib.request
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


def read(Network):
    Lower_Black = np.array([0, 0, 0])
    Upper_Black = np.array([100, 100, 100])
    center = None
    img = cv2.imread('Sample_Number.jpg',1)
    img = cv2.resize(img, (500, 500))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    mask_inrange = cv2.inRange(hsv, Lower_Black, Upper_Black)
    mask_morph_open = cv2.morphologyEx(mask_inrange, cv2.MORPH_OPEN, kernel)
    mask_dilate = cv2.dilate(mask_morph_open, kernel, iterations=2)
    mask = mask_dilate
    res = cv2.bitwise_and(img, img, mask=mask)
    array_digits_data = []
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('grey', img)
    # cv2.imshow("inrange", mask_inrange)
    # #cv2.imshow("erode", mask_erode)
    # cv2.imshow("morph open", mask_morph_open)
    # #cv2.imshow("morph closed", mask_morph_closed)
    # # cv2.imshow("mask dilate", mask_dilate)
    # # cv2.imshow("res", res)

    contours, heir = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    print("Number of contours: " + str(len(contours)))
    # for x in contours:
    #     print(x, "contour")
    #cv2.drawContours(img, contours, -1, [255, 0, 0], 3)
    cv2.imshow('grey', img)

    if (len(contours) >= 1):
        for digit in contours:
            num = 0
            filename = 'EXAMPLE_OUTPUT' + str(num) + '.jpg'
            x, y, h, w = cv2.boundingRect(np.float32(digit))
            digit_image = img[y:y + w, x:x + h]
            gray_digit_image = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
            gray_digit_image = cv2.resize(gray_digit_image, (28, 28))
            digit_data = np.array(gray_digit_image)
            digit_data = digit_data.astype('float32').flatten()/ 255.0
            digit_data_new = digit_data.reshape(784,1)
            cv2.imwrite(filename, gray_digit_image)
            array_digits_data.append(digit_data_new)
            cv2.imshow("Final Image111", gray_digit_image)
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
            num += 1
            print(Network.feedforward(digit_data_new).shape, "output")
            print(np.argmax(Network.feedforward(digit_data_new)), "Output")

    cv2.imshow("Final Image", img)
    cv2.waitKey(0)
    cv2.destroyWindow()
    return(array_digits_data)


# cap = cv2.VideoCapture(0)
# Lower_Black = np.array([0, 0, 0])
# Upper_Black = np.array([80, 80, 80])
# center = None
#
# while(True):
#     ret, img = cap.read()
#     img = cv2.flip(img, 1)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     kernel = np.ones((3, 3), np.uint8)
#     mask_inrange = cv2.inRange(hsv, Lower_Black, Upper_Black)
#     mask_morph_open = cv2.morphologyEx(mask_inrange, cv2.MORPH_OPEN, kernel)
#     mask_dilate = cv2.dilate(mask_morph_open, kernel, iterations=2)
#     mask = mask_dilate
#     res = cv2.bitwise_and(img, img, mask=mask)
#     # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#
#     contours, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print("Number of contours: " + str(len(contours)))
#     cv2.drawContours(img, contours, -1, [255, 0, 0], 3)
#     cv2.imshow('grey', img)
#
#     if (len(contours) >= 1):
#         cnt = max(contours, key = cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(np.float32(cnt))
#         cv2.circle(img, center, 5, (0, 0, 255), -1)
#         left_corner_x, left_corner_y = int(int(x) - radius), int(int(y) - radius)
#         right_corner_x, right_corner_y = int(int(x) + radius), int(int(y) + radius)
#         # print(left_corner_x, "left corner_x")
#         # print(left_corner_y, "left corner_y")
#         # print(right_corner_x, "left corner_x")
#         # print(right_corner_y, "left corner_y")
#         cv2.rectangle(img, (left_corner_x,left_corner_y), (right_corner_x, right_corner_y), (0,0,0), 2)
#         # print((x,y))
#         # print(radius)
#
#     cv2.imshow('IPWebcam', img)
#     cv2.imshow("Final Image", img)
#     time.sleep(.1)
# cv2.waitKey(0)
# cv2.destroyWindow()





# URL = "http://10.0.0.198:8080"
#
# while True:
#     img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
#     img = cv2.imdecode(img_arr, -1)
#     print(img)
#    # cv2.imshow('IPWebcam', img)
#
#     if cv2.waitKey(1):
#         break


# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#
# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     out.write(frame)
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
# cv2.line(img,(0,0),(150,15),(0,255,255), 5)
# cv2.rectangle(img,(150,25),(200,150),(0,0,255),15)
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# cv2.polylines(img, [pts], True, (0,255,255), 3)
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'Sup I can write',(0,130), font, .5, (160,255,155), 2, cv2.LINE_AA)
#
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img =  cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
# px = img[155,155]
# print(px)
# img[155,155] = [255,225,255]
# px = img[155, 155]
# print(px)
#
# px = img[100:150,100:150]
# print(px)
# img[100:150,100:150] = [255,255,255]
# watch_face = img[37:111,107:194]
# img[10:84,10:97] = watch_face
#
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img1 = cv2.imread('3D-Matplotlib.png')
# img2 = cv2.imread('mainlogo.png')
# rows,cols,channels = img2.shape
# print(rows, cols)
# roi = img1[0:rows, 0:cols ]
# rio_shift = img1[100:rows+100, 100:cols+100]
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grey", img2gray)
# ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
# mask_inv = cv2.bitwise_not(mask)
# cv2.imshow("mask", mask)
# cv2.imshow("ret", ret)
# cv2.imshow("inversemask", mask_inv)
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
# cv2.imshow("img_bg", img1_bg)
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
#
# img1_bg2 = cv2.bitwise_and(rio_shift, rio_shift, mask = mask)
# img2_bg2 = cv2.bitwise_and(img2, img2, mask = mask_inv)
# dst_1 = cv2.add(img1_bg2, img2_bg2)
# img1[100:rows+100, 100:cols+100] = dst
# cv2.imshow("img2_bg2", img2_bg2)
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread('bookpage.jpg')
# retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
# cv2.imshow('original',img)
# cv2.imshow('threshold',threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# cv2.imshow('original',img)
# cv2.imshow('Adaptive threshold',th)
# cv2.waitKey(0)
# cv2.destroyAllWindows()