python train_rtog_simclr_representations.py \
  --data_dir="/export/medical_ai/ucsf" \
  --model_dir="/export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.0005" \
  --batch_size=128 \
  --epochs=20 \
  --eval_freq=5 \
  --weight_decay=1e-4 \
  --lr=0.0005 \
  --lr_schedule=cosine \
  --momentum=0.9 \
  --n_views=2 \
  --base_model=resnet50 \
  --pretrained=True \
  --out_dim=128 \
  --temperature=0.07 \
  --minimum_primary_gleason_score=4 \


