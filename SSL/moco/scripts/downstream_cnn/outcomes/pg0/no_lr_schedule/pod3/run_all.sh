#!/bin/bash
echo "Running 5 runs, no learning rate"
sh scripts/downstream_cnn/outcomes/pg0/no_lr_schedule/pod3/run1.sh
sh scripts/downstream_cnn/outcomes/pg0/no_lr_schedule/pod3/run2.sh
sh scripts/downstream_cnn/outcomes/pg0/no_lr_schedule/pod3/run3.sh
sh scripts/downstream_cnn/outcomes/pg0/no_lr_schedule/pod3/run4.sh
sh scripts/downstream_cnn/outcomes/pg0/no_lr_schedule/pod3/run5.sh
python scripts/downstream_cnn/outcomes/pg0/no_lr_schedule/pod3/collect_results.py
