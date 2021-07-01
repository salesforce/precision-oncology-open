#!/bin/bash
echo "Running 5 runs, lr *= gamma every E epochs"
sh scripts/downstream_cnn/outcomes/epoch_gamma/run1.sh
sh scripts/downstream_cnn/outcomes/epoch_gamma/run2.sh
sh scripts/downstream_cnn/outcomes/epoch_gamma/run3.sh
sh scripts/downstream_cnn/outcomes/epoch_gamma/run4.sh
sh scripts/downstream_cnn/outcomes/epoch_gamma/run5.sh
python scripts/downstream_cnn/outcomes/epoch_gamma/collect_results.py

