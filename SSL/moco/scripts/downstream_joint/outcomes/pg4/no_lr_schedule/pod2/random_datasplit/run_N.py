import os

N = 5 # Number of runs to execute
out_dir = "/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/distant_met_5year/joint_training/no_lr_schedule/pod2/"
command = """
python train_multimodal_model.py \
  --optimizer=Adam \
  --num_epochs=150 \
  --weight_decay=5e-5 \
  --learning_rate=0.0000003 \
  --schedule=constant \
  --n_input_features=128 \
  --use_cache=True \
  --max_cache_size=0 \
  --batch_size=32 \
  --study_number_path="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/features/RTOG_{}_quilts/" \
  --output_dir="/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/distant_met_5year/joint_training/no_lr_schedule/pod2/run{}" \
""".strip()
current_runs = os.listdir(out_dir)
if current_runs:
    m = max([int(r.split('run')[1]) for r in current_runs])
else:
    m = 0

for i in range(m + 1, m + N + 1):
    out_dir_run = os.path.join(out_dir, 'run{}'.format(i))
    print("Run {}".format(i))
    run_command = command.format("{}", i)
    print(run_command)
    os.system(run_command)






