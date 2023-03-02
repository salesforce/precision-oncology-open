python train_rtog_cnn_on_representations_outcome.py \
  --optimizer=Adam \
  --num_epochs=150 \
  --weight_decay=5e-5 \
  --learning_rate=0.000003 \
  --schedule=constant \
  --n_input_features=128 \
  --use_cache=True \
  --max_cache_size=0 \
  --batch_size=32 \
  --study_number_path="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/features/RTOG_{}_quilts/" \
  --output_dir="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/distant_met_5year/no_lr_schedule/pod4/run1" \