import pandas as pd
import os
import numpy as np
import argparse
import cv2

from collections import defaultdict


def cross_val_split(
    labels, inp_path, out_path, nr_folds=5, take_every_x_image=5
):
    # We want roughly the same number of pos and neg samples in each fold
    pos_participants = labels[labels["pos_pcr_test"] == 1]["Video ID"].values
    neg_participants = labels[labels["pos_pcr_test"] == 0]["Video ID"].values
    nr_pos, nr_neg = len(pos_participants), len(neg_participants)

    # permute those
    np.random.seed(42)
    permuted_pos = np.random.permutation(pos_participants)
    permuted_neg = np.random.permutation(neg_participants)

    # Assign participants to the folds
    folds = defaultdict(list)
    id_to_fold = {}
    for counter in range(max([nr_pos, nr_neg])):
        fold_index = counter % nr_folds
        if counter < len(permuted_pos):
            folds[fold_index].append(permuted_pos[counter])
            id_to_fold[permuted_pos[counter]] = fold_index

        if counter < len(permuted_neg):
            folds[fold_index].append(permuted_neg[counter])
            id_to_fold[permuted_neg[counter]] = fold_index

    # check that no id is in the dictionary twice
    # and check classes in dict
    check_labs = labels.set_index("Video ID")
    all_ids = []
    check_dict = {}
    for key, vals in folds.items():
        check_dict[key] = [check_labs.loc[v]["pos_pcr_test"] for v in vals]
        all_ids.extend(vals)
    assert len(all_ids) == len(np.unique(all_ids))

    # make directories
    label_str = {0: "negative", 1: "positive"}
    os.makedirs(out_path, exist_ok=True)
    for fold in range(nr_folds):
        os.makedirs(os.path.join(out_path, "split" + str(fold)), exist_ok=True)
        for lab in label_str.values():
            os.makedirs(
                os.path.join(out_path, "split" + str(fold), lab),
                exist_ok=True
            )

    # create id to fold mapping --> directly yields out path
    id_to_path = {}
    for key, vals in folds.items():
        for user_id in vals:
            vid_label = label_str[check_labs.loc[user_id]["pos_pcr_test"]]
            vid_fold = "split" + str(id_to_fold[user_id])
            id_to_path[user_id] = os.path.join(out_path, vid_fold, vid_label)

    # make images and sort into dir
    for test_vid in os.listdir(inp_path):
        user_id = test_vid[:4]
        if user_id not in id_to_path:
            print(user_id, "not in labels", test_vid)
            continue
        vid_save_path = id_to_path[user_id]

        print("----------")
        print(test_vid)
        print("to be saved in", vid_save_path)

        capture = cv2.VideoCapture(os.path.join(inp_path, test_vid))

        count = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if count % take_every_x_image == 0:
                cv2.imwrite(
                    os.path.join(
                        vid_save_path, f"{test_vid.split('.')[0]}_{count}.jpg"
                    ), frame
                )
            count += 1

    # print number of files per folder
    for fold in range(nr_folds):
        for lab in label_str.values():
            print(
                fold, lab, "--> nr files:",
                len(
                    os.listdir(
                        os.path.join(out_path, "split" + str(fold), lab)
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--nr_folds", required=True, type=int, help="Path to model"
    )
    parser.add_argument(
        "-d",
        "--data_path",
        required=True,
        help="basic path to drive with all data"
    )
    parser.add_argument("-c", "--cropped_dir_name", default="cropped_videos")
    parser.add_argument("-o", "--out_dir_name", default="cross_validation")
    args = parser.parse_args()

    data_path = args.data_path

    # path with the cropped videos
    inp_path = os.path.join(data_path, args.cropped_dir_name)
    # path where to output the cross validation split data
    out_path = os.path.join(data_path, args.out_dir_name)

    # read labels
    labels = pd.read_csv(os.path.join(data_path, "raw", "labels.csv"))

    # run
    cross_val_split(labels, inp_path, out_path, nr_folds=args.nr_folds)
