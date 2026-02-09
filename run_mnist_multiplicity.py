import os
import subprocess

from rashomon_cap_test import calculate_rashomon_capacity

#Evaluate the global rashomon set and then evaluates 10 times the individual rashomon sets

# cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 1.0 --dataset_name mnist --fed_dir training_data/mnist/  --wandb True --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --run_name global --project_name Mnist_20_clients_evaluation_new  --baseline_accuracy 0.92167 --baseline_model training_data/mnist/5_0.25_1/"
# pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                         shell=True, preexec_fn=os.setsid)
# pro4.wait()
# calculate_rashomon_capacity(path="training_data/mnist/", num_clients=20, run_name="rc_global", project_name="Mnist_20_clients_evaluation_new", dataset_name="mnist")


for i in range(20):
    cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 0.05 --dataset_name mnist --fed_dir training_data/mnist/  --wandb True --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --run_name individual_{i} --project_name Mnist_20_clients_evaluation_new  --baseline_accuracy 0.92167  --baseline_model training_data/mnist/5_0.25_1/"
    pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            shell=True, preexec_fn=os.setsid)
    pro4.wait()
    calculate_rashomon_capacity(path="training_data/mnist/", num_clients=1, run_name=f"rc_{i}_individual",
                                project_name="Mnist_20_clients_evaluation_new", dataset_name="mnist")

# # Thresholds evaluation, (you only need 60% of the clients to be in the epsilon-ball)

for t in [0.6, 0.75, 0.9]:
    cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 20 --threshold {t} --dataset_name mnist --fed_dir training_data/mnist/  --wandb True --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --run_name {t}_global --project_name Mnist_20_clients_evaluation_new  --baseline_accuracy 0.92167 --baseline_model training_data/mnist/5_0.25_1/"
    pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            shell=True, preexec_fn=os.setsid)
    pro4.wait()
    calculate_rashomon_capacity(path="training_data/mnist/", num_clients=20, run_name=f"rc_{t}_global",
                                project_name="Mnist_20_clients_evaluation_new", dataset_name="mnist")








# cmd = f"python  multiplicity_evaluate.py --num_clients 50 --sampled_test_nodes_per_round 1.0 --dataset_name mnist --fed_dir training_data/mnist/  --wandb True --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --run_name global --project_name Mnist_50_clients_evaluation_new  --baseline_accuracy 0.92736  --baseline_accuracy 0.90787 --baseline_model training_data/mnist/8_1.0_3/"
# pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                         shell=True, preexec_fn=os.setsid)
# pro4.wait()
# # calculate_rashomon_capacity(path="training_data/mnist/", num_clients=20, run_name="rc_global", project_name="Mnist_20_clients_evaluation_new", dataset_name="mnist")


# for i in range(50):
#     cmd = f"python  multiplicity_evaluate.py --num_clients 50 --sampled_test_nodes_per_round 0.05 --dataset_name mnist --fed_dir training_data/mnist/  --wandb True --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --run_name individual_{i} --project_name Mnist_50_clients_evaluation_new  --baseline_accuracy 0.90787 --baseline_model training_data/mnist/8_1.0_3/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     # calculate_rashomon_capacity(path="training_data/mnist/", num_clients=1, run_name=f"rc_{i}_individual",
#     #                             project_name="Mnist_50_clients_evaluation_new", dataset_name="mnist")

# # Thresholds evaluation, (you only need 60% of the clients to be in the epsilon-ball)

# for t in [0.6, 0.75, 0.9]:
#     cmd = f"python  multiplicity_evaluate.py --num_clients 50 --sampled_test_nodes_per_round 50 --threshold {t} --dataset_name mnist --fed_dir training_data/mnist/  --wandb True --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --run_name {t}_global --project_name Mnist_50_clients_evaluation_new  --baseline_accuracy 0.90787 --baseline_model training_data/mnist/8_1.0_3/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     # calculate_rashomon_capacity(path="training_data/mnist/", num_clients=50, run_name=f"rc_{t}_global",
#     #                             project_name="Mnist_50_clients_evaluation_new", dataset_name="mnist")










