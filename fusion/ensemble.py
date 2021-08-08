import argparse
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc


def main(args):

    results_csvs = []

    # loop over configs splits
    for config in args.config: 
        # loop over cv splits
        for split in range(1,6):
            for n in range(1,args.top_n+1):

                cmd = subprocess.run(["python", "run.py", "--config", config, "--test", "--cv_split", str(split), "--top_n", str(n)], capture_output=True)
                stdout = cmd.stdout.decode()
                try: 
                    results_csv = [a for a in stdout.split('\n') if a .startswith('RESULTS_PATH')][0].split()[1]
                    results_csvs.append(results_csv)

                except: 
                    print(f'ERROR on: config: {config} | split: {split} | top_n: {n}')
                print(results_csv)

    df = pd.read_csv(results_csvs[0])
    df = df.set_index('id')
    y = df[['y']]
    df = df[['prob']]

    for idx, csv in enumerate(results_csvs[1:]): 
        df_next = pd.read_csv(csv).set_index('id')
        df_next = df_next[['prob']]
        df = df.join(df_next, rsuffix=f'_{idx}', how='outer')

    df = df.mean(axis=1).to_frame()
    df.columns = ['ensemble_prob']
    df = df.join(y)

    auroc = roc_auc_score(df['y'], df['ensemble_prob'])
    auprc = average_precision_score(df['y'], df['ensemble_prob'])
    print(f'AUROC: {auroc:.3f}')
    print(f'AUPRC: {auprc:.3f}')


    # plot
    fpr, tpr, _ = roc_curve(df['y'], df['ensemble_prob'])
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('tabnet_ensemble_supervised_auroc.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs='+',
        type=str,
        help="paths to base config")
    parser.add_argument(
        "--top_n",
        type=int,
        default=1,
        help="Use top N for ensemble") 
    args = parser.parse_args() 
    main(args)
