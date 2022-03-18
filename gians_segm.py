import logging
import os
import time
from venv import logger

import cv2
import numpy as np

f1 = "I0_test.jpg"
f2 = "I0_bad_contour.jpg"
f3 = "I_calaresu.jpg"
f4 = "I_lullia.jpg"
f5 = "I1_test-crop.jpg"
f6 = "I2_test-crop.jpg"
f7 = "I_bicolor.jpg"
fname = f7

cyan_lower = np.array([34, 85, 30])
cyan_upper = np.array([180, 252, 234])
white_lower = np.array([0, 0, 255])
white_upper = np.array([180, 255, 255])
green_lower = np.array([1, 0, 0])
green_upper = np.array([80, 255, 255])

contour_color = (0, 255, 0)  # green contour (BGR)
fill_color = list(contour_color)


def measure_area(image_rgb, color_rgb):
    # image_rgb = np.array(Image.open("p.png").convert('RGB'))

    # Work out what we are looking for
    # color = [254, 254, 254]

    # Find all pixels where the 3 RGB values match "color", and count them
    result = np.count_nonzero(np.all(image_rgb == color_rgb, axis=2))
    return result


def anonimize(image):
    _, w, _ = image.shape
    x, y, w, h = 0, 0, w, 40
    # Draw black background rectangle
    cv2.rectangle(image, (x, x), (x + w, y + h), (0, 0, 0), -1)
    return image


def put_text(image, text):
    # blue = (209, 80, 0, 255),  # font color
    white = (255, 255, 255, 255)  # font color
    x, y, w, h = 10, 40, 20, 40
    # Draw black background rectangle
    cv2.rectangle(image, (x, x), (x + w, y + h), (0, 0, 0), -1)
    cv2.putText(
        image,  # numpy array on which text is written
        text,  # text
        (x, y),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        1,  # font size
        white,  # font color
        1)  # font stroke
    return image


def all_contours(finput):
    img = cv2.imread(finput)
    img = anonimize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img


def contours_gray(finput):
    img = cv2.imread(finput)
    # convert img to grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = anonimize(img_grey)
    cv2.imshow('contours_gray ***', img_grey)
    # set a thresh
    thresh = 200
    # get threshold image
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create an empty image for contours
    img_contours = np.zeros(img.shape)
    # draw the contours on the empty image
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    return img_contours


def search_best_mask(finput):
    # NB: this routine work only on the black background
    #
    img = cv2.imread(finput)
    img = anonimize(img)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # upper_color = np.array([179, 71, 89]) # RGB 25 89 89
    # lower_color = np.array([178, 100, 11])  # RGB 0 23 24

    cnt_max = 0
    h_max_best = 180
    s_max_best = 253
    v_max_best = 0
    h_min_best = 0
    s_min_best = 0
    v_min_best = 0
    h_max = h_max_best
    s_max = s_max_best
    i = 0
    step = 16
    n_tot = 256 * 256 * 256 * 256 / step
    time_start = time.perf_counter()
    for h_min in range(0, 255, step):
        for s_min in range(0, 255, step):
            for v_min in range(0, 255, step):
                for v_max in range(0, 255, step):
                    if h_max >= h_min and s_max >= s_min and v_max >= v_min:
                        lower_color = np.array([h_min, s_min, v_min])
                        upper_color = np.array([h_max, s_max, v_max])
                        mask_color = cv2.inRange(imghsv, lower_color, upper_color)
                        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if cnt_max < len(contours):
                            cnt_max = len(contours)
                            # h_max_best = h_max
                            # s_max_best = s_max
                            v_max_best = v_max
                            h_min_best = h_min
                            s_min_best = s_min
                            v_min_best = v_min
                    i += 1
                    remaining = n_tot - i
                    if i % 1000 == 0:
                        time_stop = time.perf_counter()
                        print(
                            f"Number of contours: {cnt_max}. Time elapsed: {(time_stop - time_start) / 60:0.1f} minutes")
                        print("done: " + str(i) + "/" + str(n_tot) + " - remaining: " + str(remaining))
                        print(str(h_min_best) + ", " + str(s_min_best) + ", " + str(s_min_best) + ", " +
                              str(h_max_best) + ", " + str(s_max_best) + ", " + str(v_max_best))
    print("Best mask found.")
    return


def contours_with_colour(finput, lower_color, upper_color):
    contour_color = (0, 255, 0)
    thick = 5
    img = cv2.imread(finput)
    img = anonimize(img)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # cyan
    # lower_color = np.array([34, 85, 30])
    # upper_color = np.array([180, 252, 234])

    mask_color = cv2.inRange(imghsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # for c in contours:
    #    cv2.drawContours(img, [c], -1, contour_color, thick)
    # im = np.copy(img)
    cv2.drawContours(img, contours, -1, contour_color, thick)
    return img


def contours_with_colour_and_fill(finput, lower_color, upper_color):
    contour_color = (0, 255, 0)
    fill_color = [0, 255, 0]
    thick = 1

    img = cv2.imread(finput)
    img = anonimize(img)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # Close contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find outer contour and fill them
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(img, pts=cnts, color=fill_color)
    cv2.drawContours(img, cnts, -1, contour_color, thick)
    return img


def show_hue(finput):
    img = cv2.imread(finput)
    img = anonimize(img)
    hue = img[:, :, 0]
    cv2.imshow('hue ***', hue)
    # cv2.imshow('rescaled hue ***', (hue - np.min(hue)) / (np.max(hue) - np.min(hue)) * 255)
    return


def fill_triangle():
    fill_color = [0, 255, 0]
    # create image
    img = cv2.imread(fname)
    img = anonimize(img)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define an array of endpoints of triangle
    points = np.array([[160, 130], [350, 130], [250, 300]])
    # Use fillPoly() function and give input as
    # image, end points,color of polygon
    cv2.fillPoly(imghsv, pts=[points], color=fill_color)
    # Displaying the image
    cv2.imshow("Triangle", imghsv)
    cv2.waitKey(0)
    # Closing all open windows
    cv2.destroyAllWindows()
    return


def contours_with_colour_and_fill_2(finput, lower_color, upper_color, contour_color, fill_color):
    thick = 1

    img = cv2.imread(finput)
    img = anonimize(img)
    area = measure_area(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), fill_color)
    logger.info("Area1=" + str(area))
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # Close contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img2 = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find outer contour and fill them
    cnts, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(img2, pts=cnts, color=fill_color)
    cv2.drawContours(img2, cnts, -1, contour_color, thick)
    return img2


def gians_main():
    global cyan_lower
    global cyan_upper
    global white_lower
    global white_upper

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')
    logger.info("Program started. Reading image from file...")

    #search_best_mask(fname)
    #cv2.waitKey()

    # show_hue(fname)

    '''
    image = all_contours(fname)
    fn, fext = os.path.splitext(os.path.basename(fname))
    cv2.imwrite(fn + "_A.jpg", image)
    cv2.imshow('All Contours', image)
    '''

    '''
    image = contours_gray(fname)
    fn, fext = os.path.splitext(os.path.basename(fname))
    cv2.imwrite(fn + "_C.jpg", image)
    cv2.imshow('Contour grayed ***', image)
    '''

    # test fill area
    fill_triangle()

    # test measure area
    # Open image and make into numpy array
    image = np.array(cv2.imread(fname))
    image = anonimize(image)
    print(type(image))
    area = measure_area(image, [255, 255, 255])
    put_text(image, "Area=" + str(area) + " px")
    fn, fext = os.path.splitext(os.path.basename(fname))
    cv2.imwrite(fn + "_area.jpg", image)
    cv2.imshow('Area in pixels ***', image)

    image = contours_with_colour(fname, cyan_lower, cyan_upper)
    fn, fext = os.path.splitext(os.path.basename(fname))
    cv2.imwrite(fn + "_B.jpg", image)
    cv2.imshow('Contour cyan ***', image)

    image = contours_with_colour_and_fill_2(fname, cyan_lower, cyan_upper, contour_color, fill_color)
    fn, fext = os.path.splitext(os.path.basename(fname))

    # Open image and make into numpy array
    # image = np.array(cv2.imread(fname))
    print(type(image))
    area = 0
    # area = measure_area(image, fill_color) # TODO THIS LINE THROWS AN ERROR
    put_text(image, "Area=TODO " + str(area) + " px")

    cv2.imwrite(fn + "_D.jpg", image)
    cv2.imshow('Contour cyan Filled ***', image)

    image = contours_with_colour_and_fill_2(fname, white_lower, white_upper, contour_color, fill_color)
    fn, fext = os.path.splitext(os.path.basename(fname))
    cv2.imwrite(fn + "_B2.jpg", image)
    cv2.imshow('Contour white filled ***', image)
    cv2.waitKey()

    logger.info("Program terminated correctly.")

def fill_contours_white_interative():
    lower_color = white_lower
    upper_color = white_upper

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    # thick=3 NOT GOOD! thick=2 THE BEST BUT NEED SECOND PASS BECAUSE OF DISCONNECTIONS!
    contour_thick = 2
    fn, fext = os.path.splitext(os.path.basename(fname)) #TODO DEBUG

    img_orig = cv2.imread(fname)
    img = img_orig.copy()
    (img_h, img_w) = img_orig.shape[:2]
    logger.info("Image loaded: " + str(img_w) + "x" + str(img_h))

    cv2.imshow('Original Image', img)
    cv2.imwrite(fn + "_DEBUG0.png", img,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # TODO: suppress artifact "A" (you need two input files: with/without trace)

    # change color space and set color mask
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # Close contour
    # ksize=(3,3,) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow('Contours', img_close_contours)
    cv2.imwrite(fn + "_DEBUG1.png", img_close_contours,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # RGB image black
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR image black
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # sharpen contours: change all non-black pixels to "fill_color"
    img_green_seg = img_filled.copy()
    # # non_black_pixels_mask = np.any(img_green_seg != [0, 0, 0], axis=-1)
    # # other way to do the same:
    black_pixels_mask = np.all(img_green_seg == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    img_green_seg[non_black_pixels_mask] = [0, 255, 0]

    # PASS TWO: Close contour
    imghsv = cv2.cvtColor(img_green_seg, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS TWO: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS THREE: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS THREE: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS FOUR: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS FOUR: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS FIVE: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS FOUR: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS LAST: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # PASS LAST: erosion
    kernel_erosion = np.ones((5, 5), np.uint8)
    # using the OpenCV erode command to morphologically process the images that user wants to modify
    img_filled = cv2.erode(img_filled, kernel_erosion, iterations=1)

    # Measure the area
    area = measure_area(cv2.cvtColor(img_filled, cv2.COLOR_BGR2RGB), fill_color)
    put_text(img_filled, "Area=" + str(area) + " px")

    # Save file with text
    # cv2.imwrite(fn + "_DEBUG14.jpg", img_filled, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(fn + "_DEBUG14.png", img_filled,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imshow('Image filled', img_filled)
    cv2.waitKey()


def fill_contours_cyan_interative():
    lower_color = cyan_lower
    upper_color = cyan_upper

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    # thick=3 NOT GOOD! thick=2 THE BEST BUT NEED SECOND PASS BECAUSE OF DISCONNECTIONS!
    contour_thick = 2
    fn, fext = os.path.splitext(os.path.basename(fname)) #TODO DEBUG

    img_orig = cv2.imread(fname)
    img = img_orig.copy()
    (img_h, img_w) = img_orig.shape[:2]
    logger.info("Image loaded: " + str(img_w) + "x" + str(img_h))

    cv2.imshow('Original Image', img)
    cv2.imwrite(fn + "_DEBUG0.png", img,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # TODO: suppress artifact "A" (you need two input files: with/without trace)

    # change color space and set color mask
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # Close contour
    # ksize=(3,3,) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imshow('Contours', img_close_contours)
    cv2.imwrite(fn + "_DEBUG1.png", img_close_contours,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    # Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # RGB image black
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR image black
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # sharpen contours: change all non-black pixels to "fill_color"
    img_green_seg = img_filled.copy()
    # # non_black_pixels_mask = np.any(img_green_seg != [0, 0, 0], axis=-1)
    # # other way to do the same:
    black_pixels_mask = np.all(img_green_seg == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    img_green_seg[non_black_pixels_mask] = [0, 255, 0]

    # PASS TWO: Close contour
    imghsv = cv2.cvtColor(img_green_seg, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS TWO: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS THREE: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS THREE: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # # PASS FOUR: Close contour
    # imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    # mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # # iteration=2 NOT GOOD!
    # img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    # # PASS FOUR: Find outer contours
    # cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    # cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # # PASS FIVE: Close contour
    # imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    # mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # # iteration=2 NOT GOOD!
    # img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    # # PASS FOUR: Find outer contours
    # cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    # cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS LAST: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # PASS LAST: erosion
    kernel_erosion = np.ones((5, 5), np.uint8)
    # using the OpenCV erode command to morphologically process the images that user wants to modify
    img_filled = cv2.erode(img_filled, kernel_erosion, iterations=1)

    # Measure the area
    area = measure_area(cv2.cvtColor(img_filled, cv2.COLOR_BGR2RGB), fill_color)
    put_text(img_filled, "Area=" + str(area) + " px")

    # Save file with text
    # cv2.imwrite(fn + "_DEBUG14.jpg", img_filled, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(fn + "_DEBUG14.png", img_filled,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imshow('Image filled', img_filled)
    cv2.waitKey()


def segmentation_sharpening_DEBUG():
    # THIS GIVES BIGGER CONTOURS BECAUSE STARTS FROM THE JPEG FILE
    img_green_seg = cv2.imread("I_bicolor_DEBUG3.jpg")
    (img_h, img_w) = img_green_seg.shape[:2]
    # loop over the image, pixel by pixel TODO improve with bitwise masks
    for y in range(0, img_h):
        for x in range(0, img_w):
            if np.any(img_green_seg[y, x] != 0):
                img_green_seg[y][x] = [0, 0, 255]
            else:
                img_green_seg[y][x] = [255, 0, 0]
    cv2.imwrite("I_bicolor_DEBUG4B.jpg", img_green_seg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite("I_bicolor_DEBUG4B.png", img_green_seg, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imshow('non black pixel to Green ***', img_green_seg)
    return


def fill_contours_white(finput):
    lower_color = white_lower
    upper_color = white_upper

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    # thick=3 NOT GOOD! thick=2 THE BEST BUT NEED SECOND PASS BECAUSE OF DISCONNECTIONS!
    contour_thick = 2
    #fn, fext = os.path.splitext(os.path.basename(finput))

    img_orig = cv2.imread(finput)
    img = img_orig.copy()
    (img_h, img_w) = img_orig.shape[:2]
    logger.info("Image loaded: " + str(img_w) + "x" + str(img_h))

    # change color space and set color mask
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # PASS 1: Close contour
    # ksize=(3,3,) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS 1: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # RGB image black
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS 1: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR image black
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # sharpen contours: change all non-black pixels to "fill_color"
    img_green_seg = img_filled.copy()
    # # non_black_pixels_mask = np.any(img_green_seg != [0, 0, 0], axis=-1)
    # # other way to do the same:
    black_pixels_mask = np.all(img_green_seg == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    img_green_seg[non_black_pixels_mask] = [0, 255, 0]

    # PASS TWO: Close contour
    imghsv = cv2.cvtColor(img_green_seg, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS TWO: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS THREE: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS THREE: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS FOUR: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS FOUR: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS FIVE: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS FIVE: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS LAST: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # PASS LAST: erosion
    kernel_erosion = np.ones((5, 5), np.uint8)
    # using the OpenCV erode command to morphologically process the images that user wants to modify
    img_filled = cv2.erode(img_filled, kernel_erosion, iterations=1)
    return img_filled


def fill_contours_cyan(finput):
    lower_color = cyan_lower
    upper_color = cyan_upper

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    # thick=3 NOT GOOD! thick=2 THE BEST BUT NEED SECOND PASS BECAUSE OF DISCONNECTIONS!
    contour_thick = 2

    img_orig = cv2.imread(finput)
    img = img_orig.copy()
    (img_h, img_w) = img_orig.shape[:2]
    logger.info("Image loaded: " + str(img_w) + "x" + str(img_h))

    # change color space and set color mask
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_color = cv2.inRange(imghsv, lower_color, upper_color)

    # PASS 1: Close contour
    # ksize=(3,3,) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS 1: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # RGB image black
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS 1: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR image black
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # sharpen contours: change all non-black pixels to "fill_color"
    img_green_seg = img_filled.copy()
    # # non_black_pixels_mask = np.any(img_green_seg != [0, 0, 0], axis=-1)
    # # other way to do the same:
    black_pixels_mask = np.all(img_green_seg == [0, 0, 0], axis=-1)
    non_black_pixels_mask = ~black_pixels_mask
    img_green_seg[non_black_pixels_mask] = [0, 255, 0]

    # PASS TWO: Close contour
    imghsv = cv2.cvtColor(img_green_seg, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS TWO: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS THREE: Close contour
    imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # iteration=2 NOT GOOD!
    img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)

    # PASS THREE: Find outer contours
    cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # # PASS FOUR: Close contour
    # imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    # mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # # iteration=2 NOT GOOD!
    # img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    # # PASS FOUR: Find outer contours
    # cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    # cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # # PASS FIVE: Close contour
    # imghsv = cv2.cvtColor(img_contours, cv2.COLOR_BGR2HSV)
    # mask_green = cv2.inRange(imghsv, green_lower, green_upper)
    # # ksize=(3,3) more disconnections; ksize=(5,5) THE BEST; ksize=(7,7) too bigger border
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # # iteration=2 NOT GOOD!
    # img_close_contours = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    # # PASS FOUR: Find outer contours
    # cnts, _ = cv2.findContours(img_close_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img_contours = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    # cv2.drawContours(img_contours, cnts, -1, contour_color, contour_thick)

    # PASS LAST: fill contours
    img_filled = np.zeros((img.shape[0], img.shape[1], 3), dtype="uint8")  # BGR black image
    cv2.fillPoly(img_filled, pts=cnts, color=fill_color)

    # PASS LAST: erosion
    kernel_erosion = np.ones((5, 5), np.uint8)
    # using the OpenCV erode command to morphologically process the images that user wants to modify
    img_filled = cv2.erode(img_filled, kernel_erosion, iterations=1)
    return img_filled

def fill_contours_all_files():
    input_directory = './images_input'
    # output_directory = './images_output'
    output_directory = input_directory
    ext = ('.jpg', '.jpeg', '.png')
    for fname in os.listdir(input_directory):
        if fname.endswith(ext):
            fn, fext = os.path.splitext(os.path.basename(fname))

            # white contours
            img_filled = fill_contours_white(input_directory + "/" + fname)
            # # Measure the area
            # area = measure_area(cv2.cvtColor(img_filled, cv2.COLOR_BGR2RGB), fill_color)
            # put_text(img_filled, "Area=" + str(area) + " px")
            # Save file with text
            cv2.imwrite(output_directory + "/" + fn + "_white" + fext, img_filled, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            img_filled = fill_contours_cyan(input_directory + "/" + fname)
            cv2.imwrite(output_directory + "/" + fn + "_cyan" + fext, img_filled, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return
