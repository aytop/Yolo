from PIL import Image
import numpy as np
import pathlib


np.set_printoptions(threshold=np.nan)


def parse_annotations(ann_path, to):
    with open(ann_path) as anns:
        pathlib.Path(to).mkdir(parents=True, exist_ok=True)
        with open(to+"/paths.txt", 'w') as logs:
            for line in anns:
                tokens = line.split(',')
                img_path = tokens[0]
                num_tok = len(tokens)
                label = np.zeros((45, 9, 9))
                for i in range(1, num_tok, 5):
                    label = label + generate_box(img_path, eval(tokens[i]), eval(tokens[i+1]), eval(tokens[i+2])
                                                         , eval(tokens[i+3]), eval(tokens[i+4]) % 10, (144, 72))
                logs.write(to+"/" + img_path.split("/")[1].split(".")[0] + "\n")
                np.save(to+"/"+img_path.split("/")[1].split(".")[0], label)


def generate_box(path, x, y, width, height, digit, size):
    anchor = (int(height / width) - 1) % 3
    anchor = anchor*15
    label = np.zeros((45, 9, 9))
    img = Image.open(path)
    (i_width, i_height) = img.size
    new_w, new_h = size
    w, h = new_w / i_width, new_h / i_height
    x, width = x * w, width * w
    y, height = y * h, height * h
    (region_width, region_height) = (new_w/9, new_h/9)
    x_center = x + width / 2
    y_center = y + height / 2
    reg = (int(x_center / region_width), int(y_center / region_height))
    if reg[1] == 9:
        reg = (reg[0],8)
    if label[anchor + 14][reg] == 1:
        for i in range(3):
            if label[i*15 + 14][reg] == 0:
                anchor = i*15
    label[anchor + digit][reg] = 1
    label[anchor + 10][reg] = x_center % region_width
    label[anchor + 11][reg] = y_center % region_height
    label[anchor + 12][reg] = width/new_w
    label[anchor + 13][reg] = height/new_h
    label[anchor + 14][reg] = 1
    return label


def check_regions(l):
    for i in range(9):
        for j in range(9):
            count = 0
            for k in range(10):
                count += l[k, i, j]
                if count > 1:
                    return False
    return True


if __name__ == '__main__':
    parse_annotations('train_annotations.csv', 'anc_numpy')
    parse_annotations('test_annotations.csv', 'anc_numpy_test')