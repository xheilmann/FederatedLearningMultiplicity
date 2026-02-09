
import os
import subprocess

from rashomon_cap_test import calculate_rashomon_capacity

cmd = "python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 1.0 --dataset_name income --fed_dir training_data/income/  --wandb True --€dataset_path data/income_reduced/ --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name global --project_name IncomeMultiplicityEvaluation --baseline_accuracy 0.75881 --baseline_model training_data/income/4_0.75_2"
pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        shell=True, preexec_fn=os.setsid)
pro4.wait()

calculate_rashomon_capacity(path="training_data/income/", num_clients=20, run_name="rc_global", project_name="IncomeMultiplicityEvaluation", dataset_name="income")


for i in range(10):
    cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 0.05 --dataset_name income --fed_dir training_data/income/  --wandb True --dataset_path data/income_reduced/ --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name individual_{i} --project_name IncomeMultiplicityEvaluation --baseline_accuracy 0.75881 --baseline_model training_data/income/4_0.75_2"
    pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            shell=True, preexec_fn=os.setsid)
    pro4.wait()
    calculate_rashomon_capacity(path="training_data/income/", num_clients=1, run_name=f"rc_{i}_individual",
                                project_name="IncomeMultiplicityEvaluation", dataset_name="income")

# for t in [0.6, 0.75, 0.9]:

#     cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 20 --threshold {t} --dataset_name income --fed_dir training_data/income/  --wandb True --dataset_path data/income_reduced/ --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name {t}_global --project_name IncomeMultiplicityEvaluation  --baseline_accuracy 0.75881 --baseline_model training_data/income/4_0.75_2"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/income/", num_clients=20, run_name=f"rc_{t}_global",
#                                 project_name="IncomeMultiplicityEvaluation", dataset_name="income")



