python train_multimodal_model.py \
  --optimizer=Adam \
  --num_epochs=150 \
  --weight_decay=5e-5 \
  --learning_rate=0.0000006 \
  --schedule=constant \
  --n_input_features=128 \
  --use_cache=True \
  --max_cache_size=0 \
  --batch_size=32 \
  --study_number_path="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained/features/RTOG_{}_quilts/" \
  --output_dir="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained/distant_met_5year/joint_training/no_lr_schedule/pod1/run1" \
  --datasplit_file="/export/medical_ai/ucsf/clinical_datasplits/dm5yr_0.20pertrial_seed3_nccn=0.677.pkl" \
