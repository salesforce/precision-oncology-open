#!/bin/bash
echo "Running 5 runs, no learning rate"
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod1/run1.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod1/run2.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod1/run3.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod1/run4.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod1/run5.sh
python scripts/downstream_cnn/outcomes/no_lr_schedule/pod1/collect_results.py
