import os
import numpy as np
import cv2
import argparse
import skvideo.io as io


def crop_videos(base_path, out_dir_name):
    raw_vid_path = os.path.join(base_path, "raw")
    out_path = os.path.join(base_path, out_dir_name)
    os.makedirs(out_path)

    # INPUT: raw vid files
    in_path_vidA = os.path.join(raw_vid_path, "videos", "Tablet A")
    vid_files_in_A = [f for f in os.listdir(in_path_vidA) if f[0] != "."]

    in_path_vidC = os.path.join(raw_vid_path, "videos", "Tablet C")
    vid_files_in_C = [f for f in os.listdir(in_path_vidC) if f[0] != "."]

    all_vids = vid_files_in_A + vid_files_in_C
    path_to_vid = ["A" for _ in range(len(vid_files_in_A))
                   ] + ["C" for _ in range(len(vid_files_in_C))]

    # Iterate over videos
    for vid_name, path_char in zip(all_vids, path_to_vid):
        # get the files for current patient
        patient_id = vid_name.split("_")[1][:4]
        path_patient = os.path.join(
            raw_vid_path, "videos", f"Tablet {path_char}", vid_name
        )
        files_per_patient = [
            f for f in os.listdir(path_patient) if f[0] != "."
        ]

        # for each of the patient's files
        for vid_file_name in files_per_patient:
            # read video
            in_path = os.path.join(path_patient, vid_file_name)

            capture = cv2.VideoCapture(in_path)
            video_arr = []
            video_ind = 0
            while True:
                # capture frame-by-frame from video file
                ret, frame = capture.read()
                if not ret:
                    break
                # CROP
                video_ind += 1
                if path_char == "C":
                    video_arr.append(frame[250:600, 460:810])
                elif path_char == "A":
                    video_arr.append(frame[250:600, 330:680])
                else:
                    raise ValueError
            video_arr = np.array(video_arr)
            capture.release()

            # WRITE video
            out_file_name = patient_id + "_" + vid_file_name.split(".")[0]
            io.vwrite(
                os.path.join(out_path, out_file_name + ".mpeg"),
                video_arr,
                outputdict={"-vcodec": "mpeg2video"}
            )
            print("cropped video saved:", out_file_name, video_arr.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        required=True,
        help="basic path to drive with all data"
    )
    parser.add_argument(
        "-o", "--out_cropped_dir_name", default="cropped_videos"
    )
    args = parser.parse_args()

    crop_videos(args.data_path, args.out_cropped_dir_name)
