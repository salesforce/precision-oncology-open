"""Convert rtog-formatted csv into a training list: slide_id, patient_idx
This is for self-supervised training on the task of predicting the patient
that a slide comes from.

E.g. output:
    77875 0
    77876 0
    77877 0
    77878 1
    77879 1
    ...

    77889 10
    77890 10
    77891 10

Usage:
    python create_trainable_list_patientID.py \
            --rtog-file="/export/medical_ai/ucsf/RTOG-9202/9.14.2020/9.14.2020 DeIDed Slide Information.xlsx" \
            --out-file="/tmp/train.csv" \

"""
import pandas as pd

import csv
import argparse
from collections import Counter


def parse_input_arguments():
    """
    Parse input arguments and set parameters/settings for training
    Returns:
        args (argparse.parse_args): Returns an arguments object with all model settings
    """
    parser = argparse.ArgumentParser(description='Convert rtog-formatted csv to trainable list of data (X, y)')
    parser.add_argument('--rtog-csv', type=str, default="/export/home/data/ucsf/RTOG-9202/9.14.2020/rtog_9202.csv", metavar='DD', help='rtog-formatted file')
    parser.add_argument('--out-csv', type=str, default="/tmp/train.csv", metavar='DD', help='pytorch-formatted trainable file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_input_arguments()
    data = pd.read_csv(args.rtog_csv, delimiter=",")
    print("Original RTOG")
    print(data)

    # The USI column contains the patient de-identifer as XXXX-XXXX
    data['Patient deID'] = list(map(lambda x: x.split('-')[0], data['USI']))

    # Eliminate Patients that only have 1 slide
    data['Patient Occurence'] = data.groupby('Patient deID')['Patient deID'].transform('size')
    data = data[data['Patient Occurence'] > 1]

    # Create labels ('Patient Index') for each entry.
    data = data.sort_values(by=['Patient deID'])
    pid2idx = {p : i for i,p in enumerate(sorted(set(data['Patient deID'])))}
    data['Patient Index'] = [pid2idx[p] for p in data['Patient deID']]
    print("Processed RTOG")
    print(data)

    # Write to CSV. SVS filenames are written as strings.
    print("Creating {} from {}".format(args.out_csv, args.rtog_csv))
    data = data.filter(["Image ID", "Patient Index"])
    data = data.rename(columns={"Image ID" : "image", "Patient Index" : "label"})
    data["image"] = list(map(str, data["image"]))
    data.to_csv(args.out_csv, index=False, header=True, sep=",")
    print(data)
