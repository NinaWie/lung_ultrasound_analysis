from cProfile import label
import pandas as pd
import os
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.metrics import accuracy_score

from covlus.metrics import sensitivity_specificity
from pocovidnet.evaluate_covid19 import Evaluator

CLASS_MAPPING = ['covid', 'pneumonia', 'regular', 'uninformative']


def predict_videos(model, input_path, process_freq=5, out_path=None):

    res_dict = {}

    for test_vid in os.listdir(input_path):
        capture = cv2.VideoCapture(os.path.join(input_path, test_vid))

        count = 0
        while True:
            ret, frame = capture.read()
            count += 1
            if not ret or count > 60:
                break

            if count % process_freq == 0:
                print("processing", test_vid, count)
                out_logits = model(frame, preprocess=True)
                print(np.around(np.exp(out_logits), 2))
                res_dict[test_vid.split(".")[0] + "_" +
                         str(count)] = np.exp(out_logits)
        capture.release()

    # convert numpy arrays
    rm_np = lambda arr: [float(v) for v in arr]
    res_dict_new = {key: rm_np(res_dict[key]) for key in res_dict.keys()}
    # save file
    if out_path is not None:
        with open(os.path.join(out_path, "prediction.json"), "w") as outfile:
            json.dump(res_dict_new, outfile)
    return res_dict_new


def get_majority(x):
    # helper function: get majority voting of label
    uni, counts = np.unique(x, return_counts=True)
    return uni[np.argmax(counts)]


def get_any(x):
    # helper function: return 0 if any is 0
    if any(x.values == 0):
        return 0
    else:
        return 2


def eval_predictions(res_dict, labels, out_path):
    # extract only the label column
    label_dict = labels["pos_pcr_test"].to_dict()

    # Make dataframe with results and ground truth
    res_df = []
    for key in res_dict.keys():
        subject = key.split("_")[0]
        if subject not in label_dict:
            # print(subject)
            continue
        gt = label_dict[subject]
        # argmax of logits
        pred_ind = np.argmax(res_dict[key])

        res_df.append(
            {
                "subject": subject,
                "file": key,
                "pred_index": pred_ind,
                "cov_prob": res_dict[key][0],
                "healthy_prob": res_dict[key][2],
                "gt": gt
            }
        )
    res_df = pd.DataFrame(res_df)
    res_df = res_df.sort_values(["subject", "file"])

    # group by subject, agg predictions by majority voting
    grouped_by_subject = res_df.groupby("subject").agg(
        {
            "cov_prob": "mean",
            "healthy_prob": "mean",
            "gt": "mean",
            "pred_index": get_majority
        }
    )
    # plot cov probability
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=grouped_by_subject, x="gt", y="cov_prob")
    plt.xlabel("Ground truth class (1=covid)", fontsize=15)
    plt.ylabel("Logit prob for COVID", fontsize=15)
    plt.savefig(os.path.join(args.out_path, "class_prob_violin.pdf"))

    # Accuracy dependent on cutoff
    print("Accuracy (video level)")
    for cutoff in np.arange(0.1, 1, 0.05):
        covid_pred = res_df["cov_prob"].values > cutoff
        print(
            "cutoff:", round(cutoff, 2), "Accuracy:",
            accuracy_score(covid_pred, res_df["gt"].values)
        )

    print("Accuracy (subject-level)")
    for cutoff in np.arange(0.1, 1, 0.05):
        covid_pred = (grouped_by_subject["cov_prob"].values >
                      cutoff).astype(int)
        print(
            round(cutoff, 2), "Accuracy:",
            round(
                accuracy_score(covid_pred, grouped_by_subject["gt"].values), 2
            )
        )
    print(
        "Label distribution gt",
        np.unique(res_df["gt"].values, return_counts=True)
    )
    print(
        "Label distribution pred",
        np.unique(res_df["pred_index"].values, return_counts=True)
    )

    # print final results with sensitivity and specificity
    res_df["vid_file"] = res_df["file"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    grouped_by_vid_file = res_df.groupby("vid_file").agg(
        {
            "cov_prob": "mean",
            "subject": "first",
            "healthy_prob": "mean",
            "gt": "mean",
            "pred_index": get_any  # TODO: any suitable?
        }
    )
    grouped_by_subject_2 = grouped_by_vid_file.groupby("subject").agg(
        {
            "cov_prob": "mean",
            "healthy_prob": "mean",
            "gt": "mean",
            "pred_index": get_any  # TODO: any suitable?
        }
    )
    sensitivity_specificity(
        (res_df["pred_index"] == 0).astype(int),
        res_df["gt"].values,
        title="all (index) - frames"
    )
    sensitivity_specificity(
        (grouped_by_vid_file["pred_index"] == 0).astype(int),
        grouped_by_vid_file["gt"].values,
        title="all (index) - vid files"
    )
    sensitivity_specificity(
        (grouped_by_subject["pred_index"] == 0).astype(int),
        grouped_by_subject["gt"].values,
        title="all (index) - subjects"
    )
    sensitivity_specificity(
        (grouped_by_subject_2["pred_index"] == 0).astype(int),
        grouped_by_subject_2["gt"].values,
        title="all (index) - subjects 2"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", required=True, help="Path to model"
    )
    parser.add_argument(
        "--model_name", default="vgg_base", help="model factory id"
    )
    parser.add_argument("-i", "--inp_vid_path", default="cropped_videos")
    parser.add_argument("-l", "--inp_label_path", required=True)
    parser.add_argument("-o", "--out_path", default="eval_results")
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # load model in evaluator class
    model = Evaluator(
        weights_dir=args.model_path,
        ensemble=True,
        num_classes=4,
        model_id=args.model_name
    )
    # create out path
    os.makedirs(args.out_path, exist_ok=True)

    res_dict = predict_videos(
        model, args.inp_vid_path, 5, out_path=args.out_path
    )
    # load results instead
    # with open("../results_vgg_cam_1.json", "r") as infile:
    #     res_dict = json.load(infile)

    # read labels
    labels = pd.read_csv(args.inp_label_path, index_col="Video ID")

    eval_predictions(res_dict, labels, out_path=args.out_path)
