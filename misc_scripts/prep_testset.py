import pickle

clinical_data = pickle.load(open("/export/medical_ai/ucsf/test_sets/testset.pkl", "rb"))
train_df = clinical_data["df_X_full"].loc[clinical_data["df_X_full"].test_set == 0]

slide_df_path = "/export/home/rtog_dfs/"
train_slides = {}
for dataset in [9408, 9910, 9202, 9413, "0126"]:
    train_slides["RTOG-" + str(dataset)] = []
    study_slide_df = pd.read_csv(slide_df_path + str(dataset) + ".csv")
    sub_train_df = train_df.loc[train_df.sn == int(dataset)]
    for cn in sub_train_df.cn_deidentified:
        patient_df = study_slide_df.loc[study_slide_df.cn_deidentified == cn]
        train_slides["RTOG-" + str(dataset)] += list(patient_df.image_id)

with open("/export/home/rtog_testset/testset.pkl", "wb") as f:
    pickle.dump(train_slides, f)

