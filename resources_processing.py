import cv2
import numpy as npy


def get_frames_from_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret_val, frame = capture.read()
        if not ret_val:
            break
        frames.append(frame)

    capture.release()
    return frames


def print_frames(frames):
    for index, frame in enumerate(frames):
        print_single_frame(index, frame)
        if cv2.waitKey(5) & 0xFF == ord('s'):  # little wait time until next iteration
            break
    cv2.destroyAllWindows()


def print_single_frame(index, frame):
    cv2.putText(frame, str(index + 1), (10, 30), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 220, 220),
                thickness=2)
    cv2.imshow('Frame ', frame)


def get_grayscaled_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def get_binary_frame(frame):
    # binary_image = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    ret, binary_image = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
    return binary_image


def invert_frame(frame):
    return 255 - frame


def dilate(frame, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return cv2.dilate(frame, kernel, iterations=1)


def erode(frame, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return cv2.erode(frame, kernel, iterations=1)


def scale(frame):
    return frame / 255


def detect_lines(frame):
    edges = cv2.Canny(frame, 160, 210, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * npy.pi / 180, threshold=140, minLineLength=100, maxLineGap=10)

    acctual_lines = []
    flags = [False] * len(lines)
    for i, line in enumerate(lines):
        if flags[i] is True:
            continue
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        for j, line_compare_to in enumerate(lines):
            if flags[j] is True:
                continue
            x1_cmp, y1_cmp, x2_cmp, y2_cmp = line_compare_to[0]
            slope_cmp = (y2_cmp - y1_cmp) / (x2_cmp - x1_cmp)
            diff = round(slope, 2) - round(slope_cmp, 2)
            if i != j and -0.03 < diff < 0.03:
                flags[i] = flags[j] = True
                if y1 > y1_cmp:
                    acctual_lines.append(line)
                else:
                    acctual_lines.append(line_compare_to)
                break
    for i, flag in enumerate(flags):
        if flag is False:
            acctual_lines.append(lines[i])

    if acctual_lines[0][0][0] < acctual_lines[1][0][0]:
        return acctual_lines[0][0], acctual_lines[1][0]
    else:
        return acctual_lines[1][0], acctual_lines[0][0]


def make_region(frame, x, y, w, h):
    region = frame[y:y + h + 1, x:x + w + 1]
    region = invert_frame(region)
    region = cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)
    region = scale(region)
    return region
