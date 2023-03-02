python train_rtog_cnn_on_representations.py \
  --optimizer=SGD \
  --num_epochs=200 \
  --weight_decay=5e-4 \
  --learning_rate=5e-07 \
  --n_input_features=128 \
  --use_cache=True \
  --max_cache_size=0 \
  --batch_size=32 \
  --base_model=MicroModel \
  --study_number_path="/export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.0005/features/RTOG_{}_quilts/" \
  --output_dir="/export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.0005/micromodel/run5" \
