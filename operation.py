import os
import sys
import shutil
import mat4py
import argparse

from PIL import Image
from utils.utils import Convert_abcd_to_xywh

# At first,put your rootpath of the WIDER_FACE dataset.
# In order to train conveniently,I move all the train and valid picutres in the data folder.


def Dataset_txt_Create(dataset_path, base_path, txt_path):
    with open(txt_path, 'w') as fp:
        # filter out those txt files.
        file_list = [x for x in os.listdir(dataset_path) if os.path.splitext(x)[1] != ".txt"]
        for filename in file_list:
            filepath = os.path.join(base_path, filename)
            fp.write(filepath + '\n')


def WIDER_Convert(dataset_path, out_path):
    # Constructing train dataset and valid dataset at the same time.

    train_dataset_path = os.path.join(dataset_path, "WIDER_train")
    valid_dataset_path = os.path.join(dataset_path, "WIDER_valid")

    train_out_path = os.path.join(out_path, "train")
    valid_out_path = os.path.join(out_path, "valid")

    task_split_info_folder = os.path.join(dataset_path, "wider_face_split")
    train_split_info_path = os.path.join(task_split_info_folder, "wider_face_train_bbx_gt.txt")
    valid_split_info_path = os.path.join(task_split_info_folder, "wider_face_valid_bbx_gt.txt")

    # train dataset make up.
    print("train dataset make up...")
    WIDER_Task_Convert(train_dataset_path, train_out_path, train_split_info_path)
    print("train dataset construct down...")

    # valid dataset make up.
    print("valid dataset make up...")
    WIDER_Task_Convert(valid_dataset_path, valid_out_path, valid_split_info_path)
    print("valid dataset construct down...")

    # train.txt make up.
    train_txt_path = os.path.join(out_path, "train.txt")
    Dataset_txt_Create(train_out_path, "face_data/train", train_txt_path)

    # valid.txt make up.
    valid_txt_path = os.path.join(out_path, "valid.txt")
    Dataset_txt_Create(valid_out_path, "face_data/valid", valid_txt_path)


def WIDER_Task_Convert(task_dataset_path, task_out_path, task_split_info_path):
    
    if not os.path.exists(task_out_path):
        os.makedirs(task_out_path, 0o777)
    
    fp = open(task_split_info_path, "r")

    index = 0
    while(True):
        image_name = fp.readline().strip()
        if(image_name==""):
            break
        index += 1

        print(f"{image_name} processing...")

        # Copy image from source path to destination path.
        img_src_path = os.path.join(task_dataset_path, "images", image_name)
        img_dst_path = os.path.join(task_out_path, f"{index}.jpg")
        if not os.path.exists(img_dst_path):
            shutil.copy(img_src_path, img_dst_path)
        
        # Parse annotation and get yolo format annotation.
        annotation_dst_path = os.path.join(task_out_path, f"{index}.txt")
        with open(annotation_dst_path, 'w') as fp_label:
            # Get the number of detection targets or the number of ground truth bounding boxes.
            num_bboxes = int(fp.readline())
            num_bboxes = num_bboxes if num_bboxes != 0 else 1
            w, h = Image.open(img_src_path).size
            for bbox_i in range(num_bboxes):
                bbox = fp.readline().strip().split(' ')
                bbox = [int(x) for x in bbox]
                # Convert to [x_center, y_center, width, height] and Normalization.
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                bbox_w   = bbox[2] / w
                bbox_h   = bbox[3] / h
                # Because there is only one class (face), so the class number is always 0.
                fp_label.write("0 {} {} {} {}\n".format(x_center, y_center, bbox_w, bbox_h))

        print(f"{image_name} process down...")


def Hand_Convert(dataset_path, out_path):
    # Constructing train dataset and valid dataset at the same time.

    train_dataset_path = os.path.join(dataset_path, "training_dataset", "training_data")
    valid_dataset_path = os.path.join(dataset_path, "validation_dataset", "validation_data")

    train_out_path = os.path.join(out_path, "train")
    valid_out_path = os.path.join(out_path, "valid")

    # train dataset make up.
    print("train dataset make up...")
    Hand_Task_Convert(train_dataset_path, train_out_path)
    print("train dataset construct down...")

    # valid dataset make up.
    print("valid dataset make up...")
    Hand_Task_Convert(valid_dataset_path, valid_out_path)
    print("valid dataset construct down...")

    # train.txt make up.
    train_txt_path = os.path.join(out_path, "train.txt")
    Dataset_txt_Create(train_out_path, "hand_data/train", train_txt_path)

    # valid.txt make up.
    valid_txt_path = os.path.join(out_path, "valid.txt")
    Dataset_txt_Create(valid_out_path, "hand_data/valid", valid_txt_path)


def Hand_Task_Convert(task_dataset_path, task_out_path):

    if not os.path.exists(task_out_path):
        os.makedirs(task_out_path, 0o777)

    image_folder = os.path.join(task_dataset_path, "images")
    annotation_folder = os.path.join(task_dataset_path, "annotations")

    image_list = [x for x in os.listdir(image_folder) if "DS_Store" not in x]
    annotation_list = [x for x in os.listdir(annotation_folder) if "DS_Store" not in x]

    index = 0
    for image_name, annotation_name in zip(image_list, annotation_list):
        index += 1

        # Annotation must match the image.
        image_prefix = os.path.splitext(image_name)[0]
        annotation_prefix = os.path.splitext(annotation_name)[0]
        assert image_prefix == annotation_prefix, \
            f"Annotation {image_prefix} doesn't match image {annotation_prefix}."

        image_src_path = os.path.join(image_folder, image_name)
        annotation_src_path = os.path.join(annotation_folder, annotation_name)

        print(f"{image_name} processing...")
        
        # image process.
        image_dst_path = os.path.join(task_out_path, f"{index}.jpg")
        if not os.path.exists(image_dst_path):
            shutil.copy(image_src_path, image_dst_path)
        
        # annotation process.
        annotation_dst_path = os.path.join(task_out_path, f"{index}.txt")
        with open(annotation_dst_path, 'w') as fp:
            w, h = Image.open(image_src_path).size
            annotation = mat4py.loadmat(annotation_src_path)
            bboxes = annotation["boxes"]
            bboxes = bboxes if type(bboxes) == list else [bboxes]
            for bbox in bboxes:
                # Convert [(xa, ya), (xb, yb), (xc, yc), (xd, yd)] to (x_center, y_center, bbox_width, bbox_height).
                x_center, y_center, bbox_w, bbox_h = Convert_abcd_to_xywh(bbox)
                # Normalization.
                x_center = x_center / w
                y_center = y_center / h
                bbox_w = bbox_w / w
                bbox_h = bbox_h / h
                # Because there is only one class (hand), so the class number is always 0.
                fp.write("0 {} {} {} {}\n".format(x_center, y_center, bbox_w, bbox_h))

        print(f"{image_name} process down...")


def main(args):
    if args.dataset_name == "face":
        WIDER_Convert(args.src_path, args.out_path)
    elif args.dataset_name == "hand":
        Hand_Convert(args.src_path, args.out_path)
    else:
        assert False, "Dataset name can't be {}.".format(args.dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_path", type=str, default=None, help="The path of dataset root path.")
    parser.add_argument("--out_path", type=str, default=None, help="The output path of this constructed dataset.")

    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset which is needed to process.")
    
    args = parser.parse_args()
    
    main(args)
