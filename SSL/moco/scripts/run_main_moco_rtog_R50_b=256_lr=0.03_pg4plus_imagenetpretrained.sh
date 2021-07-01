python main_moco_rtog.py \
  --arch=resnet50 \
  --lr=0.03 \
  --batch-size 256 \
  --mlp \
  --moco-t 0.2 \
  --aug-plus \
  --cos \
  --dist-url='tcp://localhost:10001' \
  --multiprocessing-distributed \
  --world-size=1 \
  --rank 0 \
  --model_dir="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained" \
  --imagenet_pretrained \
  --minimum_primary_gleason_score=4 \
  --resume="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained/checkpoint_0033.pth.tar" \
  "/export/medical_ai/ucsf"

# --pretrained="./pretrained_models/moco_v2_800ep_pretrain.pth.tar" \
