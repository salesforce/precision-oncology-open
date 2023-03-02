python train_rtog_simclr_representations.py \
  --data_dir="/export/medical_ai/ucsf" \
  --model_dir="/export/medical_ai/ucsf/simclr_rtog/model_resnet34_gp4plus" \
  --batch_size=128 \
  --epochs=80 \
  --eval_freq=5 \
  --weight_decay=1e-4 \
  --lr=0.05 \
  --lr_schedule=cosine \
  --momentum=0.9 \
  --n_views=2 \
  --base_model=resnet34 \
  --pretrained=False \
  --out_dim=128 \
  --temperature=0.07 \
  --minimum_primary_gleason_score=4 \


