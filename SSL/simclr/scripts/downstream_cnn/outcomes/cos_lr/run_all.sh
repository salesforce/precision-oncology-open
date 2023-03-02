#!/bin/bash
echo "Running 5 runs, cosine learning rate"
sh scripts/downstream_cnn/outcomes/cos_lr/run1.sh
sh scripts/downstream_cnn/outcomes/cos_lr/run2.sh
sh scripts/downstream_cnn/outcomes/cos_lr/run3.sh
sh scripts/downstream_cnn/outcomes/cos_lr/run4.sh
sh scripts/downstream_cnn/outcomes/cos_lr/run5.sh
python scripts/downstream_cnn/outcomes/cos_lr/collect_results.py
