import os
import numpy as np

dir = "/export/medical_ai/ucsf/ssl_rtog/moco/model_R50_b=256_lr=0.03_pg4plus/distant_met_5year/no_lr_schedule/pod1"
run_dirs = ['run1', 'run2', 'run3', 'run4', 'run5']


def extract_auc(dir_contents, prefix="testing"):
    paths = [f for f in dir_contents if f[-4:] == ".png" and f.split('_')[0].lower() == prefix.lower()]
    auc_vals = [float(t.split('=')[-1].split('.png')[0]) for t in paths]
    return np.array(auc_vals)


fnames = []
for d in run_dirs:
    fnames.append(os.listdir(os.path.join(dir, d)))
test_aucs = np.vstack([extract_auc(dir_contents) for dir_contents in fnames])
print(test_aucs.shape)
print("{} test auc across {}".format(dir, run_dirs))
print("max: {}".format(np.max(test_aucs)))
print("mean of maxes: {}".format(np.mean(np.max(test_aucs, axis=1))))

