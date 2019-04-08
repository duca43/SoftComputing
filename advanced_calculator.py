from builtins import enumerate
import cv2
import math
import neural_network as nn
import resources_processing as res_pro
from number_object import NumberObject
from collections import Counter


def main():
    print('Program started...')

    # Setup and train neural network
    model = nn.get_trained_neural_network()

    # go through all videos, calculate sum and write it in file
    with open('out.txt', 'r+') as file:
        lines = file.readlines()
        lines[0] = "RA 105/2015 Dusan Maksic\n"
        for i in range(0, 10):
            frames = res_pro.get_frames_from_video('video-' + str(i) + '.avi')
            sum_of_number = calculate_sum_for_video(frames, model)
            # print(sum_of_number)
            lines[i + 2] = 'video-' + str(i) + '.avi\t' + str(sum_of_number) + '\n'
        file.truncate(0)
        file.seek(0)
        file.write(''.join(lines))

    cv2.destroyAllWindows()
    print('Program finished...')


def calculate_sum_for_video(frames, model):
    # Line detection
    green_line, blue_line = res_pro.detect_lines(res_pro.get_grayscaled_frame(frames[0]))

    detected_numbers = []
    sum_of_numbers = 0
    for idx, frame in enumerate(frames):
        original_frame = frame.copy()
        frame = res_pro.get_grayscaled_frame(frame)

        # Draw detected lines
        cv2.line(original_frame, (green_line[0], green_line[1]), (green_line[2], green_line[3]), (0, 0, 255), 2)
        cv2.line(original_frame, (blue_line[0], blue_line[1]), (blue_line[2], blue_line[3]), (0, 0, 255), 2)

        # Transform image for easier ROI detecion
        frame = res_pro.get_binary_frame(frame)
        frame = res_pro.invert_frame(frame)
        frame = res_pro.erode(frame, (3, 3))

        # Every detected object initially is not updated
        for num_index, number_object in enumerate(detected_numbers):
            number_object.updated = False
            detected_numbers[num_index] = number_object

        # Extract contours
        img, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            x, y, w, h = cv2.boundingRect(contour)
            if 8 < radius < 30:
                # Draw enclosing circle
                # cv2.circle(original_frame, center, radius, (0, 255, 0), 2)

                # Extract detected number from frame
                region = res_pro.make_region(frame, x, y, w, h)
                value = nn.do_prediction_for_region(region, model)

                already_detected = False
                for num_index, number_object in enumerate(detected_numbers):
                    distance_from_object = distance(center, number_object.center)
                    if distance_from_object < 20:
                        already_detected = True
                        number_object.center = center
                        number_object.radius = radius
                        number_object.values.append(value)
                        number_object.updated = True
                        number_object.frames_disappeared = 0
                        detected_numbers[num_index] = number_object
                        break

                if already_detected is False:
                    new_number_object = NumberObject(center, radius, value)
                    detected_numbers.append(new_number_object)

        for num_index, number_object in enumerate(detected_numbers):
            if number_object.updated is False:
                number_object.frames_disappeared += 1
                if number_object.frames_disappeared > NumberObject.max_frames_disappeared:
                    del detected_numbers[num_index]

        # Line equatations for both green and blue line
        ag, bg, cg = line_from_points((green_line[0], green_line[1]), (green_line[2], green_line[3]))
        ab, bb, cb = line_from_points((blue_line[0], blue_line[1]), (blue_line[2], blue_line[3]))

        for num_index, number_object in enumerate(detected_numbers):
            dist_green = (ag * number_object.center[0] + bg * number_object.center[1] + cg) / math.sqrt(
                ag * ag + bg * bg)
            dist_blue = (ab * number_object.center[0] + bb * number_object.center[1] + cb) / math.sqrt(
                ab * ab + bb * bb)

            # cv2.line(original_frame, (number_object.center[0] + number_object.radius, number_object.center[1]), (green_line[0], green_line[1]), (200, 220, 13), 2)
            # cv2.line(original_frame, (number_object.center[0] - number_object.radius, number_object.center[1]), (green_line[2], green_line[3]), (50, 150, 222), 2)

            cv2.circle(original_frame, number_object.center, number_object.radius, (0, 255, 0), 2)
            if 0 < dist_green - number_object.radius < 4 and number_object.collided_green is False \
                    and (number_object.center[0] + number_object.radius) >= green_line[0] \
                    and (number_object.center[0] - number_object.radius) <= green_line[2]:
                # print(idx + 1, 'collision GREEN', most_common_element(number_object.values))
                number_object.collided_green = True
                detected_numbers[num_index] = number_object
                sum_of_numbers -= most_common_element(number_object.values)
            elif 0 < dist_blue - number_object.radius < 4 and number_object.collided_blue is False \
                    and (number_object.center[0] + number_object.radius) >= blue_line[0] \
                    and (number_object.center[0] - number_object.radius) <= blue_line[2]:
                # print(idx + 1, 'collision BLUE', most_common_element(number_object.values))
                number_object.collided_blue = True
                detected_numbers[num_index] = number_object
                sum_of_numbers += most_common_element(number_object.values)
        res_pro.print_single_frame(idx, original_frame)
        if cv2.waitKey(5) & 0xFF == ord('s'):  # little wait time until next iteration
            break
    return sum_of_numbers


def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def line_from_points(x, y):
    a = y[1] - x[1]
    b = x[0] - y[0]
    c = -(a * (x[0]) + b * (x[1]))
    return a, b, c


def most_common_element(list_of_values):
    data = Counter(list_of_values)
    return max(list_of_values, key=data.get)


main()
