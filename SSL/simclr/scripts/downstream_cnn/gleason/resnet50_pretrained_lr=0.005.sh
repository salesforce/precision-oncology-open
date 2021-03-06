python train_rtog_cnn_on_representations.py \
  --optimizer=Adam \
  --num_epochs=50 \
  --weight_decay=5e-2 \
  --learning_rate=0.00001 \
  --n_input_features=128 \
  --use_cache=True \
  --max_cache_size=0 \
  --batch_size=32 \
  --study_number_path="/export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.005/features/RTOG_{}_quilts/" \
  --output_dir="/export/medical_ai/ucsf/simclr_rtog/imagenet_data/model_resnet50_gp4plus_pretrained_lr=0.005/" \
