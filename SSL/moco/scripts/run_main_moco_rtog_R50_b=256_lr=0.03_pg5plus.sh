python main_moco_rtog.py \
  --arch=resnet50 \
  --lr=0.03 \
  --batch-size 256 \
  --mlp \
  --moco-t 0.2 \
  --aug-plus \
  --cos \
  --epochs 200 \
  --dist-url='tcp://localhost:10002' \
  --multiprocessing-distributed \
  --world-size=1 \
  --rank 0 \
  --model_dir="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg5plus" \
  --minimum_primary_gleason_score=5 \
  --resume="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg5plus/checkpoint_0194.pth.tar" \
  "/export/medical_ai/ucsf"

# --pretrained="./pretrained_models/moco_v2_800ep_pretrain.pth.tar" \