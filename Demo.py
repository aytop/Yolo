import Inference
import cv2


def main():
    inf = Inference.Inference()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap.open()
    while True:
        ret, img = cap.read()
        img = inf(img)
        cv2.imshow('Yolo detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()