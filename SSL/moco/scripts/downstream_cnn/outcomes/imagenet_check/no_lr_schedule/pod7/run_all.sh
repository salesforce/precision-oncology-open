#!/bin/bash
echo "Running 5 runs, no learning rate"
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod7/run1.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod7/run2.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod7/run3.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod7/run4.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod7/run5.sh
python scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod7/collect_results.py
