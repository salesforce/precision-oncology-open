#!/bin/bash
echo "Running 5 runs, no learning rate"
sh scripts/downstream_cnn/outcomes/pg5/no_lr_schedule/pod5/run1.sh
sh scripts/downstream_cnn/outcomes/pg5/no_lr_schedule/pod5/run2.sh
sh scripts/downstream_cnn/outcomes/pg5/no_lr_schedule/pod5/run3.sh
sh scripts/downstream_cnn/outcomes/pg5/no_lr_schedule/pod5/run4.sh
sh scripts/downstream_cnn/outcomes/pg5/no_lr_schedule/pod5/run5.sh
python scripts/downstream_cnn/outcomes/pg5/no_lr_schedule/pod5/collect_results.py
