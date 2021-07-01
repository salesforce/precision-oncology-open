export IMAGE_SIZE=224
export MODEL_NUM=8
export MEAN_REGULARIZE=-1
export LR=0.001
export PRE_TRAINED=True
export OVERSAMPLING=True
export NUM_SAMPLES=50
export NUM_WORKERS=8
export EPOCHS=1000
export CUTOUT=0
export HARD_NEGATIVE=False
export NUM_TEST_REPEATS=1
export DATA_DIR=/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level
export TRAIN_CSV=/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level/ssl_patient_deID_train.csv
export VAL_CSV=/export/home/data/ucsf/RTOG-9202/9.14.2020/95cpu_patches_256_patchsize_224_resize_0_overlap_2_level/ssl_patient_deID_test.csv
export MODEL_DIR=/export/home/code/metamind/precision_oncology/results/lr=${LR}_cutout=${CUTOUT}_mean_reg=${MEAN_REGULARIZE}
export LOGFILE=$MODEL_DIR/logfile.txt

mkdir -p $MODEL_DIR
touch $LOGFILE
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
#      --resume_from_last_model \
#      --resume_from_checkpoint \
#      --multigpu=True \
       2>&1 | tee -a $LOGFILE

