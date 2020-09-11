import os
import sys
import cv2
import torch
import torch.nn as nn

from torchvision import transforms

import argparse

from utils.utils import *
from modeling.darknet import Darknet
from modeling.yolov3_tiny import Yolov3_tiny


Wider_tiny_anchors = [6,8, 12,16, 23,30, 46,59, 102,136, 293,387]
Wider_num_classes = 1


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Antomatic constructing model implementation
    model = Darknet(args.model_config_path, args.input_size).to(device)
    
    # Manual constructing Yolov3 tiny model implementation
    # model = Yolov3_tiny(3, Wider_num_classes, Wider_tiny_anchors, args.input_size).to(device)

    if os.path.splitext(args.ckpt_path)[1] == ".weights":
        # load darknet model weights file.
        model.load_darknet_weights(args.ckpt_path)
    elif os.path.splitext(args.ckpt_path)[1] in [".pth", ".pkl", ".pt"]:
        # load pytorch model parameters file.
        model.load_state_dict(torch.load(args.ckpt_path))

    model.eval()

    classes = load_classes(args.class_path)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() and args.cuda else torch.FloatTensor

    if args.image is not None:
        img = cv2.imread(args.image)
        h, w = img.shape[:2]

        boxes = detect(model, img, (h, w), args.input_size, args.cuda, args.conf_thres, args.nms_iou_thres)

        if boxes is not None:
            for x1, y1, x2, y2  in boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 2), 2)
                x_center = (x1 + x2)/2
                if boxes.shape[0] == 1:
                    show_string = None
                    if x_center <= h/3:
                        show_string = "right"
                    elif x_center >= 2*h/3:
                        show_string = "left"
                    else:
                        show_string = "middle"
                    print(show_string)
                    cv2.putText(img, show_string, (40, 40), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1., (255, 255, 2), 2)

        cv2.imshow("show", img)
        cv2.waitKey(0)

        if args.out_filename is not None:
            cv2.imwrite(img, args.out_filename)

    elif args.video is not None:
        video_name = Convert_String_to_Int(args.video)
        
        cap = cv2.VideoCapture(video_name)
    
        while(1):
            ret, frame = cap.read()
            h, w = frame.shape[:2]

            boxes = detect(model, frame, (h, w), args.input_size, args.cuda, args.conf_thres, args.nms_iou_thres)

            if boxes is not None:
                for x1, y1, x2, y2  in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 2), 2)
                    x_center = (x1 + x2)/2
                    if boxes.shape[0] == 1:
                        show_string = None
                        if x_center <= h/3:
                            show_string = "right"
                        elif x_center >= 2*h/3:
                            show_string = "left"
                        else:
                            show_string = "middle"
                        print(show_string)
                        cv2.putText(frame, show_string, (40, 40), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1., (255, 255, 2), 2)

            cv2.imshow("show", frame)
            button = cv2.waitKey(10)
            
            # ord function can get char related ASCii code.
            # chr function is the reverse function of ord function.
            # When inputting 's' char by keyboard, the image play will stop. 
            # Using random char except 'q' char can continue playing.
            # When inputting 'q' char by kerboard, the image play will exit.
            if button == ord('s'):
                button = cv2.waitKey(0)
            
            if button == ord('q'):
                break


def detect(model, img, img_shape, input_size, cuda=True, conf_thres=0.5, nms_iou_thres=0.4):
    h, w = img_shape
    input_tensor = Convert_Img_to_Tensor(img)
    input_tensor = Pad_Tensor_to_Square(input_tensor, 0)
    input_tensor = Tensor_Resize(input_tensor, input_size)
    input_tensor = torch.unsqueeze(input_tensor, dim=0)
    
    if cuda:
        input_tensor = input_tensor.cuda()

    detections = model(input_tensor)

    detections = Non_Max_Suppression(detections, conf_thres, nms_iou_thres)[0]

    if detections is not None:
        detections = Rescale_Boxes(detections, input_size, (h, w))
        boxes = detections.numpy()
        boxes = boxes[:, :4]

        return boxes
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cuda", type=bool, default=False, help="Deciding if inferencing with gpu.")
    
    parser.add_argument("--conf_thres", type=float, default=0.8, help="The threshold to filter out bounding boxes whose confidence scores is below threshold.")
    parser.add_argument("--nms_iou_thres", type=float, default=0.4, help="The IOU threshold to filter out rebundant bounding boxes.")
    parser.add_argument("--input_size", type=int, default=416, help="The rescaling tensor size when inputting model.")

    parser.add_argument("--ckpt_path", type=str, default="config/yolov3_wider.pth", help="The path of model parameters file.")
    parser.add_argument("--class_path", type=str, default="config/wider.names", help="The path of detection classes file.")
    parser.add_argument("--model_config_path", type=str, default="config/yolov3_wider.cfg", help="The path of model definition file.")
    
    parser.add_argument("--video", type=str, default="0", help="The path of video file needed to process or camera index.")
    parser.add_argument("--image", type=str, default=None, help="The path of image file needed to process.")
    parser.add_argument("--out_filename", type=str, default=None, help="The save path of processed video or The save path of processed image.")

    args = parser.parse_args()

    main(args)