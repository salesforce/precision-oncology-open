MODEL_DIR="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained-2"
python main_moco_rtog.py \
  --arch=resnet50 \
  --lr=0.03 \
  --workers=32 \
  --epochs=800 \
  --batch-size 256 \
  --mlp \
  --moco-t 0.2 \
  --aug-plus \
  --cos \
  --dist-url='tcp://localhost:10000' \
  --multiprocessing-distributed \
  --world-size=1 \
  --rank 0 \
  --model_dir=$MODEL_DIR \
  --imagenet_pretrained \
  --minimum_primary_gleason_score=4 \
  --resume="$MODEL_DIR/checkpoint_0537.pth.tar" \
  "/export/medical_ai/ucsf/tissue_vs_non_pkl/v2"

# --pretrained="./pretrained_models/moco_v2_800ep_pretrain.pth.tar" \
