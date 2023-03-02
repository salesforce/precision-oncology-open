export DATA_DIR=/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level
export IMAGE_SIZE=224
export MODEL_NUM=8
export MEAN_REGULARIZE=-1
export MODEL_DIR=/export/home/code/metamind/precision_oncology/results
export LR=0.000001
export PRE_TRAINED=True
export OVERSAMPLING=True
export NUM_SAMPLES=50
export NUM_WORKERS=2
export EPOCHS=1000
export CUTOUT=25
export HARD_NEGATIVE=False
export NUM_TEST_REPEATS=3
export TRAIN_CSV=/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level/ssl_patient_deID_train.csv
export VAL_CSV=/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level/ssl_patient_deID_test.csv

python main_repeat_testing_split_multitissue.py \
       --datadir=$DATA_DIR \
       --imagesize=$IMAGE_SIZE \
       --model=$MODEL_NUM \
       --mean_regularize=$MEAN_REGULARIZE \
       --modeldir=$MODEL_DIR \
       --lr=$LR \
       --pretrained=$PRE_TRAINED \
       --oversampling=$OVERSAMPLING \
       --num_samples=$NUM_SAMPLES \
       --num_workers=$NUM_WORKERS \
       --epochs=$EPOCHS \
       --cutout=$CUTOUT \
       --hard_negative=$HARD_NEGATIVE \
       --train_csv=$TRAIN_CSV \
       --val_csv=$VAL_CSV \
       --num_test_repeats=$NUM_TEST_REPEATS \
       --resume_from_last_model \
       --resume_from_checkpoint

