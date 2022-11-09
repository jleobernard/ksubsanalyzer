import cv2

IMAGE_EXTENSIONS = ["jpg", "png"]
CSV_ANNOTATION_COL_NAMES = ["label", "x0", "y0", "x1", "y1", "filename", "width", "height"]


def read_image(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
