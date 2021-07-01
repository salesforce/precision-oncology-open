template="""
python train_rtog_cnn_on_representations.py \\
  --optimizer={} \\
  --num_epochs={} \\
  --weight_decay=5e-4 \\
  --learning_rate={} \\
  --n_input_features=128 \\
  --use_cache=True \\
  --max_cache_size=0 \\
  --batch_size=32 \\
  --base_model=MicroModel \\
  --study_number_path="/export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.0005/features/RTOG_{}_quilts/" \\
  --output_dir="/export/medical_ai/ucsf/simclr_rtog/model_resnet50_gp4plus_pretrained_lr=0.0005/micromodel/run{}" \\
""".strip()

optimizers=['Adam', 'SGD']
num_epochs=[200]
learning_rates = [1e-6, 5e-7, 1e-7, 5e-8]

i = 0
for optim in optimizers:
    for epoch in num_epochs:
        for lr in learning_rates:
            run=template.format(optim,epoch,lr, '{}', i)
            filename = 'run{}.sh'.format(i)
            with open(filename, 'w') as f:
                f.write(run)
                print("Wrote {}: optim={}, epochs={}, lr={}.".format(filename, optim, epoch, lr))
            i += 1
