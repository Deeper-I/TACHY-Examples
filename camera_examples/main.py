import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="V4L2 camera viewer")
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize output image to WIDTH HEIGHT (e.g. --resize 640 320)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.resize:
            w, h = args.resize
            frame = cv2.resize(frame, (w, h))
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
