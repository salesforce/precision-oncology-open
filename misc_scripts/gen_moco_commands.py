def gen_moco_commands(base_path, model_checkpoint):
    model_checkpoint = base_path + model_checkpoint

    output_dir = base_path + "/features/{}_features/"
    input_dir = "/export/medical_ai/ucsf/tissue_vs_non_pkl/v2/{}/"
    datasets = ["RTOG-9202", "RTOG-9413", "RTOG-9408", "RTOG-9910", "RTOG-0126"]
    base_model = "resnet50"

    command = "python3 moco_feature_generation.py --output_dir {} --input_dir {} --checkpoint_path {} --base_model {}"

    mkdirs = ["mkdir " + base_path + "features/"]
    for dataset in datasets:
        d_input_dir = input_dir.format(dataset)
        d_output_dir = output_dir.format(dataset)
        mkdirs += ["mkdir " + output_dir.format(dataset)]
        print(command.format(d_output_dir, d_input_dir, model_checkpoint, base_model))
        print()

    for x in mkdirs:
        print(x)

    print()
    print ("*" * 80)
    print()


    tissue_label_dir = "/export/medical_ai/ucsf/tissue_vs_non_pkl/v2/{}/"
    tissue_dir = "/export/medical_ai/ucsf/{}/tissue_pickles_v2/"
    slide_df = "/export/home/rtog_dfs/master_lists/{}.csv"
    save_dir = base_path + "/features/{}_quilts/"
    feature_dir = base_path+"/features/{}_features/"

    command = "python3 quilt_generation.py  --tissue_label_dir {} --slide_df_loc {} --save_dir {} --feature_dir {} --save_features"

    mkdirs = ["mkdir " + base_path + "features/"]
    for dataset in datasets:
        d_slide_df = slide_df.format(dataset.split("-")[1])
        d_save_dir = save_dir.format(dataset)
        d_feature_dir = feature_dir.format(dataset)
        d_tissue_label_dir = tissue_label_dir.format(dataset)
        d_feature_dir = feature_dir.format(dataset)

        mkdirs += ["mkdir " + save_dir.format(dataset)]
        print(command.format(d_tissue_label_dir, d_slide_df, d_save_dir, d_feature_dir))
        print()

    for x in mkdirs:
        print(x)
