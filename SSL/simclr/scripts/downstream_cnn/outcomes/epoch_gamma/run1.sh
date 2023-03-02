python train_rtog_cnn_on_representations_outcome.py \
  --optimizer=Adam \
  --num_epochs=150 \
  --weight_decay=5e-5 \
  --learning_rate=0.00001 \
  --schedule=30_0.5 \
  --n_input_features=128 \
  --use_cache=True \
  --max_cache_size=0 \
  --batch_size=32 \
  --study_number_path="/export/medical_ai/ucsf/ssl_rtog/simclr/model_resnet50_gp4plus_pretrained_lr=0.0005/features/RTOG_{}_quilts/" \
  --output_dir="/export/medical_ai/ucsf/ssl_rtog/simclr/model_resnet50_gp4plus_pretrained_lr=0.0005/distant_met_5year/epoch_gamma/run1" \
