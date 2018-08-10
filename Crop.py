from PIL import Image
import pathlib

def main():
    with open('train_annotations.csv') as ann_file:
        for line in ann_file:
            tokens = line.split(',')
            image_name = tokens[0]
            number_of_dig = len(tokens)
            for i in range(1, number_of_dig, 5):
                crop(image_name, tokens[i], tokens[i+1], tokens[i+2], tokens[i+3], tokens[i+4])


def crop(i_name, x, y, width, height, label):
    image = Image.open(i_name)
    left = eval(x)
    right = eval(x)+eval(width)
    top = eval(y)
    bottom = eval(y)+eval(height)
    print(left,top,right,bottom)
    cropped = image.crop((left, top, right, bottom))
    pathlib.Path('croped/'+label.strip()).mkdir(parents=True, exist_ok=True)
    cropped.save('croped/'+label.strip()+'/'+i_name.split('/')[1])


if __name__ == '__main__':
    main()