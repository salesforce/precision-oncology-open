#!/bin/bash
echo "Running 5 runs, no learning rate"
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod6/run1.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod6/run2.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod6/run3.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod6/run4.sh
sh scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod6/run5.sh
python scripts/downstream_cnn/outcomes/imagenet_check/no_lr_schedule/pod6/collect_results.py
