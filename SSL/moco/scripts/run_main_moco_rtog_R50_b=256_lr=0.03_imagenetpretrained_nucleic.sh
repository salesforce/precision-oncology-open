MODEL_DIR="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_imagenet_pretrained_nucleic"
python main_moco_rtog.py \
  --arch=resnet50 \
  --lr=0.03 \
  --workers=32 \
  --epochs=40 \
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
  --nucleic_density \
  --resume="$MODEL_DIR/checkpoint_0027.pth.tar" \
  "/export/medical_ai/ucsf/tissue_vs_non_pkl/nuclei_counts/v2"

# --minimum_primary_gleason_score=4 \
# --pretrained="./pretrained_models/moco_v2_800ep_pretrain.pth.tar" \

# Note: nucleic density increases 1.7M pg4 patches to 19.7M. It increases 5.16M total patches to 39.3M.
# Imagenet, with 1.6M patches trained for 200 epochs in moco paper.
# Adjust the epochs accordingly.
