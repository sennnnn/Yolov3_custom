import os
import sys
import shutil
import argparse

from PIL import Image


# At first,put your rootpath of the WIDER_FACE dataset.
# In order to train conveniently,I move all the train and valid picutres in the data folder.


def Dataset_txt_Create(dataset_path, txt_path):
    with open(txt_path, 'w') as fp:
        for filename in os.listdir(dataset_path):
            filepath = os.path.join(dataset_path, filename)
            fp.write(filepath + '\n')


def WIDER_Convert(dataset_path, out_path):
    # Constructing train dataset and valid dataset at the same time.
    task_list = ["train", "valid"]

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
    print("train dataset make up...")
    WIDER_Task_Convert(valid_dataset_path, valid_out_path, valid_split_info_path)
    print("train dataset construct down...")

    # train.txt make up.
    train_txt_path = os.path.join(out_path, "train.txt")
    Dataset_txt_Create(train_dataset_path, train_txt_path)

    # valid.txt make up.
    valid_txt_path = os.path.join(out_path, "valid.txt")
    Dataset_txt_Create(valid_dataset_path, valid_txt_path)


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
        img_dst_path = os.path.join(task_out_path, f"{index}.jpg")
        img_src_path = os.path.join(task_dataset_path, "images", image_name)
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
                width    = bbox[2] / w
                height   = bbox[3] / h
                # Because there is only one class (face), so the class number is always 0.
                fp_label.write("0 {} {} {} {}\n".format(x_center, y_center, width, height))

        print(f"{image_name} process down...")


def Hand_Convert(dataset_path, out_path):
    pass


def main(args):
    if args.dataset_name == "FACE":
        WIDER_Convert(args.WIDER_path, args.out_path)
    elif args.dataset_name == "Hand":
        Hand_Convert(arg.Hand_path, args.out_path)
    else:
        assert False, "Dataset name can't be {}.".format(args.dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--WIDER_path", type=str, default=None, help="The path of WIDER FACE dataset root path.")
    parser.add_argument("--Hand_path", type=str, default=None, help="The path of Hand dataset root path.")
    parser.add_argument("--out_path", type=str, default=None, help="The output path of this constructed dataset.")

    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset which is needed to process.")
    
    args = parser.parse_args()
    
    main(args)
