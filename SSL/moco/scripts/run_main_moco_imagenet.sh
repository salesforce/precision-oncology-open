python main_moco.py \
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
  --model_dir="/export/medical_ai/ucsf/ssl_rtog/moco/imagenet_check" \
  --resume="/export/medical_ai/ucsf/ssl_rtog/moco/imagenet_check/checkpoint_0092.pth.tar" \
  "/export/medical_ai/imagenet/ILSVRC/Data/CLS-LOC"

