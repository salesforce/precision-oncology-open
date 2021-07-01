# Preprocessing
### Patching the slides
Histhopathology slides are too big to be processed as a whole, so therefore we split them up into 256 by 256 patches. [This](https://github.com/MetaMind/precision_oncology/blob/master/save_image_patches_to_disk_parallel.py) script can be used for that, it also contains an example call.

Example usage:
```
   python save_image_patches_to_disk_parallel.py \
         --svs_dir=/export/home/data/ucsf/svs_test/ \
         --write_dir=/export/medical_ai/ucsf/tmp \
         --resize_size=224 \
         --patch_size=256 \
         --overlap=0 \
         --level=3
```

### Tissue classification
Not all of the created patches contain tissue, and some slides contain artifacts from scanning, drawing or damage. To filter the non-tissue patches, a classifier has been trained. The script to classify all patches can be found [here](https://github.com/MetaMind/precision_oncology/blob/master/artifact_classifier/generate_pickle_files.py). You need to define the input and output path in the script itself. 

Example call after adjusting the input and output paths:
```
python3 generate_pickle_files.py
```

### Create tissue pickles
Loading many small files is slower than loading one big file from disk. As we often need all tissue patches of a single slide, we save all patches of a single file in a pickle. The script for this can be found [here](https://github.com/MetaMind/precision_oncology/blob/pandanet/pandanet/scripts/save_tissue_pickles.py)

Example call
```
python3 save_tissue_pickles.py --label_dir /export/medical_ai/ucsf/tissue_vs_non_pkl/v2/RTOG-9202/ \
--output_dir /export/medical_ai/ucsf/tissue_pickles_v2/RTOG-9202/
```

# SSL
### Train MoCo
As we don't have patch level labels, we need a self supervised learning method to generate features. For this we use MoCo and the script can be fond [here](https://github.com/MetaMind/precision_oncology/blob/master/SSL/moco/main_moco_rtog.py)

The command used for the [PANDA dataset](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data) is as follows:
```
python3 main_moco_pandas.py \
--arch=resnet50 \  
--lr=0.03 \  
--batch-size 256 \  
--mlp \  
--moco-t 0.2 \  
--aug-plus \  
--cos \  
--dist-url='tcp://localhost:10001' \  
--multiprocessing-distributed \  
--world-size=1 \  
--rank 0 \  
--model_dir="/export/medical_ai/kaggle_panda/SSL_recreate/models/model_R50_b=256_lr=0.03_pg0plus" \  
"/export/medical_ai/kaggle_panda/SSL_recreate"
```

# Downstream task
### Generate features
Now that we have a trained feature generator, it's most efficient to apply it to all patches and save the features to pickles. The script for this can be found [here](https://github.com/MetaMind/precision_oncology/blob/master/SSL/moco/moco_feature_generation.py). 

Example call:
```
python3 moco_feature_generation.py \
--output_dir /export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.05_b=256/features/RTOG-9413_features/ \
--input_dir /export/medical_ai/ucsf/RTOG-9413/tissue_pickles_v2/ \
--checkpoint_path /export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.05_b=256/checkpoint-epoch18.pt \
--base_model resnet50
```

### Create feature quilts
In order to make a prediction for a patient, we need to combine all tissue of a single patient. This script first finds all separate chuncks of tissue in all slides, and then attempts to effieciently place these in a 200x200 grid, starting in the center, circling outwards. The script for this can be found [here](https://github.com/MetaMind/precision_oncology/blob/master/SSL/simclr/quilt_generation.py). The script can generate the feature quilts using feature vectors, but it can also construct the quilt from image patches, for visual inspection. To save the feature quilt, use the `--save_features` argument, to save the image quilts, use the `--save_images` argument.

This requires csv file which contains the link between the slides image_id and the cn_deidentified number, such that all slides of a single patient can be found. This csv file can be generated using the rtog_helper class, which can be found [here](https://github.com/MetaMind/precision_oncology/blob/master/clinical_data_classifier/rtog_helper.py)

Example call: 
python3 quilt_generation.py --tissue_dir /export/medical_ai/ucsf/RTOG-9202/tissue_pickles_v2/ \
--tissue_label_dir /export/medical_ai/ucsf/tissue_vs_non_pkl/v2/RTOG-9202/ \
--slide_df_loc /export/medical_ai/kaggle_panda/RTOG_pandanet_labels/RTOG-9202.csv \
--save_dir /export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus//features/RTOG-9202_quilts/ \
--feature_dir /export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/features/RTOG-9202_features/ \
--save_features


### Train downstream CNN
Now that we have a feature quilt per patient, and a patient level label, we can train a classifier. That can be done using this [script](https://github.com/MetaMind/precision_oncology/blob/master/SSL/simclr/train_rtog_cnn_on_representations_outcome.py)

The command used for Pandas is as follows:
```
python3 train_rtog_cnn_on_representations_outcome.py \
   --optimizer=Adam \
   --num_epochs=150 \
   --weight_decay=5e-5 \
   --learning_rate=0.0000007 \
   --schedule=constant \
   --n_input_features=128 \
   --use_cache=True \
   --max_cache_size=0 \
   --batch_size=32 \
   --study_number_path="/export/medical_ai/kaggle_panda/quilts{}/" \
   --output_dir="/export/medical_ai/kaggle_panda/SSL_recreate/downstream_output"
```
