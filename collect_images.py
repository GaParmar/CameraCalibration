import os
import sys
import argparse
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--path", default="data/", help="the path where images will be saved")
parser.add_argument("--seq_name", required=True, help="name for the current sequence")
parser.add_argument('--debug', help='Enable VSCode debugging', default=False, action='store_true')
parser.add_argument("--camera_id", default=1, type=int)
parser.add_argument("--length", default=30, help="in seconds")

args = parser.parse_args()

if __name__ == "__main__":
    if args.debug:
        # Ref: https://vinta.ws/code/remotely-debug-a-python-app-inside-a-docker-container-in-visual-studio-code.html
        import ptvsd
        print("Enabling attach starts.")
        ptvsd.enable_attach(address=('0.0.0.0', 8091))
        ptvsd.wait_for_attach()
        print("Enabling attach ends.")
    
    outpath = os.path.join(args.path, args.seq_name)
    images = []
    camera = cv2.VideoCapture(args.camera_id)
    start_ts = time.time()
    while time.time() < (start_ts+args.length):
        start_epoch = time.time()
        ret, image = camera.read()
        assert ret
        images.append(image)
        while time.time()<start_epoch+0.5:
            pass
    print(outpath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    [cv2.imwrite(os.path.join(outpath, f"{idx}.png"), img) for idx, img in enumerate(images)]

