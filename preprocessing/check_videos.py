import pandas as pd
import os
import numpy as np
import argparse

CASTOR_REL_PATH = os.path.join(
    "COVID-19_Lung_Ultrasound_Study_M_csv_export_20220120150424",
    "COVID-19_Lung_Ultrasound_Study_M_export_20220120.csv"
)


def check_data(raw_data_path):

    in_path_vidA = os.path.join(raw_data_path, "videos", "Tablet A")
    vid_files_in_A = [f for f in os.listdir(in_path_vidA) if f[0] != "."]

    in_path_vidC = os.path.join(raw_data_path, "videos", "Tablet C")
    vid_files_in_C = [f for f in os.listdir(in_path_vidC) if f[0] != "."]

    all_vids = vid_files_in_A + vid_files_in_C
    path_to_vid = ["A" for _ in range(len(vid_files_in_A))
                   ] + ["C" for _ in range(len(vid_files_in_C))]

    # load mapping of IDs to castor IDs
    id_mapping = pd.read_csv(
        os.path.join(raw_data_path, "key_code_images_castor.csv"),
        delimiter=";"
    ).set_index("IMAGES")
    # csv with all patient information
    df_castor = pd.read_csv(
        os.path.join(raw_data_path, CASTOR_REL_PATH), delimiter=";"
    )
    print(
        "Ratio of positive cases:", np.sum(df_castor["cov_test"].values == 0)
    )

    # Make a dictionary with the stats
    patient_dict = {}
    all_patient_ids = []
    for patient, path_char in zip(all_vids, path_to_vid):
        patient_id = patient.split("_")[1][:4]
        if patient_id in all_patient_ids:
            print("ID in db twice", patient_id)
        all_patient_ids.append(patient_id)

        path_patient = os.path.join(
            raw_data_path, "videos", f"Tablet {path_char}", patient
        )
        files_per_patient = len(
            [f for f in os.listdir(path_patient) if f[0] != "."]
        )
        patient_dict[patient_id] = {
            "nr_files": files_per_patient,
            "full_path": patient,
            "castor": id_mapping.loc[patient_id]["CASTOR"]
        }

    # Print out results
    print("---------- In mapping but not in videos / castor")
    for i in range(1, 48):
        i_string = "0" + str(i) if i < 10 else str(i)
        id_A = "A0" + i_string
        if id_A not in id_mapping.index:
            print("ID not in Castor mapping", id_A)
        if id_A not in all_patient_ids:
            print("ID not in video", id_A)

    print("------ Patients B / Tablet C --------")
    for i in range(1, 22):
        i_string = "0" + str(i) if i < 10 else str(i)
        id_B = "B0" + i_string
        if id_B not in id_mapping.index:
            print("ID not in video-Castor mapping", id_B)
        castor_id = id_mapping.loc[id_B]["CASTOR"]
        if castor_id not in df_castor["Record Id"].values:
            print("ID not in Castor", castor_id)
        if id_B not in all_patient_ids:
            print("ID not in video", id_B)

    print("--------- In castor but not in id mapping")
    for castor_id in df_castor["Record Id"].values:
        if castor_id not in id_mapping["CASTOR"].values:
            print("ID not in id mapping", castor_id)

    # save everything in a csv
    patient_info = pd.DataFrame(patient_dict).transpose().reset_index(
    ).sort_values("index")
    patient_info = patient_info.rename(
        columns={
            "nr_files": "Number of video files",
            "full_path": "Video path",
            "castor": "Castor ID",
            "index": "Video ID"
        }
    )
    patient_info = patient_info.set_index("Castor ID")
    final_summary = id_mapping.reset_index().merge(
        patient_info, how="outer", left_on="CASTOR", right_index=True
    )
    final_summary = final_summary.rename(columns={
        "CASTOR": "Castor ID"
    }).drop(columns=["IMAGES"]).set_index("Castor ID")

    final_summary.to_csv(os.path.join(raw_data_path, "data_summary.csv"))
    print(
        "Saved final summary:",
        os.path.join(raw_data_path, "data_summary.csv")
    )
    return final_summary


def save_labels(summary, raw_data_path):
    df_castor = pd.read_csv(
        os.path.join(raw_data_path, CASTOR_REL_PATH), delimiter=";"
    )
    castor_labels = df_castor[[
        'Record Id', 'cov_test', 'clin_diagn#COVID19_pneumonia',
        'clin_diagn#other_viral_pneumonia', 'clin_diagn#bacterial_pneumonia',
        'clin_diagn#other_lung_disease', 'clin_diagn#healthy_lung',
        'oth_diagn', 'vir_diagn', 'bact_diagn'
    ]]
    merged = summary.merge(
        castor_labels, left_on="Castor ID", right_on="Record Id"
    ).set_index("Video ID")
    # transform label column into dictionary
    label_dict_nans = merged["cov_test"].to_dict()
    label_dict = {
        key: int(not label_dict_nans[key])
        for key in label_dict_nans.keys()
        if (not pd.isna(key)) and (not pd.isna(label_dict_nans[key]))
    }
    df = pd.DataFrame(label_dict, index=["pos_pcr_test"]).swapaxes(1, 0)
    # merge with rest of the data
    summary_with_labels = summary.merge(
        df, left_on="Video ID", right_index=True
    )

    summary_with_labels.to_csv(
        os.path.join(raw_data_path, "labels.csv"), index=False
    )
    print("Saved labels:", os.path.join(raw_data_path, "labels.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        required=True,
        help="basic path to drive with all data"
    )
    args = parser.parse_args()

    raw_data_path = os.path.join(args.data_path, "raw")
    data_summary = check_data(raw_data_path)
    save_labels(data_summary, raw_data_path)
