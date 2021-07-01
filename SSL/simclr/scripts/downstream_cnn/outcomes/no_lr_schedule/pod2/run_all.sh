#!/bin/bash
echo "Running 5 runs, no learning rate"
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod2/run1.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod2/run2.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod2/run3.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod2/run4.sh
sh scripts/downstream_cnn/outcomes/no_lr_schedule/pod2/run5.sh
python scripts/downstream_cnn/outcomes/no_lr_schedule/pod2/collect_results.py
