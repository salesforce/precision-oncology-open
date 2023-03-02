import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='NRG Feature-Block Gleason Finetuning.')
parser.add_argument('--pod', required=True, type=str,
                    help='Option --pod missing. E.g. --pod=2')
args = parser.parse_args()

N = 5 # Number of runs to execute
out_dir = "/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained/distant_met_5year/joint_training/no_lr_schedule/pod{}/{}"
out_dir = out_dir.format(args.pod, {})
command = """
python train_multimodal_model.py \
  --optimizer=Adam \
  --num_epochs=300 \
  --weight_decay=5e-5 \
  --learning_rate={} \
  --schedule=constant \
  --n_input_features=128 \
  --use_cache=True \
  --max_cache_size=0 \
  --batch_size=32 \
  --study_number_path="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus_imagenet_pretrained/features/RTOG_{}_quilts/" \
  --output_dir="{}" \
  --datasplit_file="/export/medical_ai/ucsf/clinical_datasplits/dm5yr_0.20pertrial_seed3_nccn=0.677.pkl" \
""".strip()

for i in range(N):
    lr = 0.1 ** ((3 + np.random.uniform() * 4)) # sample between 1e-3 and 1e-7
    out_dir = out_dir.format("run_lr={}".format(lr))
    command = command.format(lr, {}, out_dir)
    if os.path.exists(out_dir):
        continue
    os.makedirs(out_dir)
    out_dir_run = os.path.join(out_dir, 'run{}'.format(i))
    print("Run {}".format(i))
    run_command = command.format("{}", i)
    print(run_command)
    os.system(run_command)






