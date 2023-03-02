#!/bin/bash
echo "Running 5 runs, no learning rate"
sh scripts/downstream_joint/outcomes/pg4/no_lr_schedule/pod4/run1.sh
sh scripts/downstream_joint/outcomes/pg4/no_lr_schedule/pod4/run2.sh
sh scripts/downstream_joint/outcomes/pg4/no_lr_schedule/pod4/run3.sh
sh scripts/downstream_joint/outcomes/pg4/no_lr_schedule/pod4/run4.sh
sh scripts/downstream_joint/outcomes/pg4/no_lr_schedule/pod4/run5.sh
python scripts/downstream_joint/outcomes/pg4/no_lr_schedule/pod4/collect_results.py
