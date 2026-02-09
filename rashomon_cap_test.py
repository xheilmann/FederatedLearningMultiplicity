import os

import numpy as np

from Utils.multiplicity_evaluation import rashomon_capacity
from main import setup_wandb


def calculate_rashomon_capacity(path: str, num_clients=2, project_name=None, run_name=None, dataset_name=None):
    wandb_run = setup_wandb(
        project_name=project_name,
        run_name=run_name,
    )

    num_models = []
    #test = os.listdir(f"Rashomon_Set")
    epsilon_dirs = {}

    # Get all epsilon directories
    for item in os.listdir(f"Rashomon_Set_{dataset_name}"):
        item_path = os.path.join(f"Rashomon_Set_{dataset_name}", item)
        if os.path.isdir(item_path) and item.startswith('epsilon_'):
            epsilon_str = item.replace('epsilon_', '')
            try:
                epsilon_val = float(epsilon_str)
                epsilon_dirs[epsilon_val] = item_path
            except ValueError:
                continue

    # Sort by epsilon values
    sorted_epsilons = sorted(epsilon_dirs.keys())

    for epsilon in sorted_epsilons:
        dir_path = epsilon_dirs[epsilon]
        num_models.append(len(os.listdir(dir_path)))


    #for each client
    rc_diffpriv_all = []
    rc_all = []
    for i in range(num_clients):
        ndarrays = np.load(f"{path}/{i}_results.npz")
        rc_diffpriv = np.zeros((len(sorted_epsilons), 1000))
        rc= np.zeros((len(sorted_epsilons), ndarrays["all_sampling_test_scores"].shape[1]))
        count = 0
        for j in range(1,len(num_models)):
            k = num_models[j]
            count += k
            if k == 0 and j>0:
                rc[j,:] = rc[j-1,:]
                rc_diffpriv[j,:] = rc_diffpriv[j-1,:]
                continue
            if k ==0 and j==0:
                continue
            scores = ndarrays[f"all_sampling_test_scores"][:count,:,:]
            rc_diffpriv[j, :], rc[j,:] = rashomon_capacity(scores)
        rc_diffpriv_all.append(rc_diffpriv)
        rc_all.append(rc)

    agg_metrics = {}

    rc = np.cumsum(np.sum(rc_diffpriv_all, axis=0),
                         axis=1)
    rc_cum = [entry / entry[-1] for entry in rc]

    agg_metrics[f"rc_90_difpriv"] = [float(np.searchsorted(entry, 0.9, "left") * 0.001) for entry in rc_cum]
    agg_metrics[f"rc_50_difpriv"] = [float(np.searchsorted(entry, 0.5) * 0.001) for entry in rc_cum]
    for entry in rc_cum:
        entry[np.isnan(entry)] = 0
    agg_metrics[f"rc_difpriv"] = rc_cum



    rc = [[e for entry in rc_all for e in entry[i]] for i in range(len(rc_all[0]))]
    agg_metrics[f"rc_90"] = np.quantile(rc, 0.9, axis=1)
    agg_metrics[f"rc_50"] = np.quantile(rc, 0.5, axis=1)

    agg_metrics[f"rc"] = rc

    wandb_run.log(agg_metrics)
    wandb_run.finish()
    
    return agg_metrics

if __name__ == "__main__":
    path = "training_data/income/"
    metrics = calculate_rashomon_capacity(path)
    print(metrics)