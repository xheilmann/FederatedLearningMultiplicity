import os
import subprocess

from rashomon_cap_test import calculate_rashomon_capacity

# # #Evaluate the global rashomon set and then evaluates 10 times the individual rashomon sets

# cmd = f"python  multiplicity_evaluate.py --num_clients 10 --sampled_test_nodes_per_round 1.0 --dataset_name dutch --fed_dir training_data/dutch_10_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name global --project_name DutchMultiplicityEvaluation_10_clients  --baseline_accuracy 0.8358  --baseline_model training_data/dutch_10_clients/7_0.25_2117271212/"
# pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                         shell=True, preexec_fn=os.setsid)
# pro4.wait()
# calculate_rashomon_capacity(path="training_data/dutch_10_clients/", num_clients=10, run_name="rc_global", project_name="DutchMultiplicityEvaluation_10_clients", dataset_name="dutch")


# for i in range(10):
#     cmd = f"python  multiplicity_evaluate.py --num_clients 10 --sampled_test_nodes_per_round 0.05 --dataset_name dutch --fed_dir training_data/dutch_10_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name individual_{i} --project_name DutchMultiplicityEvaluation_10_clients  --baseline_accuracy 0.8358  --baseline_model training_data/dutch_10_clients/7_0.25_2117271212/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_10_clients/", num_clients=1, run_name=f"rc_{i}_individual",
#                                 project_name="DutchMultiplicityEvaluation_10_clients", dataset_name="dutch")

# # # Thresholds evaluation, (you only need 60% of the clients to be in the epsilon-ball)

# for t in [0.9]:#[0.6, 0.75, 0.9]:
#     cmd = f"python  multiplicity_evaluate.py --num_clients 10 --sampled_test_nodes_per_round 10 --threshold {t} --dataset_name dutch --fed_dir training_data/dutch_10_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name {t}_global --project_name DutchMultiplicityEvaluation_10_clients  --baseline_accuracy 0.8358 --baseline_model training_data/dutch_10_clients/7_0.25_2117271212/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_10_clients/", num_clients=10, run_name=f"rc_{t}_global",
#                                 project_name="DutchMultiplicityEvaluation_10_clients", dataset_name="dutch")






# # Evaluate the global rashomon set and then evaluates 10 times the individual rashomon sets

# cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 1.0 --dataset_name dutch --fed_dir training_data/dutch_20_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name global --project_name DutchMultiplicityEvaluation_20_clients  --baseline_accuracy 0.83279 --baseline_model training_data/dutch_20_clients/5_0.5_3930982344"
# pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                         shell=True, preexec_fn=os.setsid)
# pro4.wait()
# calculate_rashomon_capacity(path="training_data/dutch_20_clients/", num_clients=20, run_name="rc_global", project_name="DutchMultiplicityEvaluation_20_clients", dataset_name="dutch")


# for i in range(10):
#     cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 0.05 --dataset_name dutch --fed_dir training_data/dutch_20_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name individual_{i} --project_name DutchMultiplicityEvaluation_20_clients  --baseline_accuracy 0.83279 --baseline_model training_data/dutch_20_clients/5_0.5_3930982344"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_20_clients/", num_clients=1, run_name=f"rc_{i}_individual",
#                                 project_name="DutchMultiplicityEvaluation_20_clients", dataset_name="dutch")

# Thresholds evaluation, (you only need 60% of the clients to be in the epsilon-ball)

# for t in [0.6, 0.75, 0.9]:
#     cmd = f"python  multiplicity_evaluate.py --num_clients 20 --sampled_test_nodes_per_round 20 --threshold {t} --dataset_name dutch --fed_dir training_data/dutch_20_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name {t}_global --project_name DutchMultiplicityEvaluation_20_clients  --baseline_accuracy 0.83279 --baseline_model training_data/dutch_20_clients/5_0.5_3930982344"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_20_clients/", num_clients=20, run_name=f"rc_{t}_global",
#                                 project_name="DutchMultiplicityEvaluation_20_clients", dataset_name="dutch")







# # # #Evaluate the global rashomon set and then evaluates 10 times the individual rashomon sets
# cmd = f"python  multiplicity_evaluate.py --num_clients 30 --sampled_test_nodes_per_round 1.0 --dataset_name dutch --fed_dir training_data/dutch_30_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name global --project_name DutchMultiplicityEvaluation_30_clients  --baseline_accuracy 0.82225 --baseline_model training_data/dutch_30_clients/5_0.5_3848704714/"
# pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                         shell=True, preexec_fn=os.setsid)
# pro4.wait()
# calculate_rashomon_capacity(path="training_data/dutch_30_clients/", num_clients=30, run_name="rc_global", project_name="DutchMultiplicityEvaluation_30_clients", dataset_name="dutch")


# for i in range(10):
#     cmd = f"python  multiplicity_evaluate.py --num_clients 30 --sampled_test_nodes_per_round 0.05 --dataset_name dutch --fed_dir training_data/dutch_30_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name individual_{i} --project_name DutchMultiplicityEvaluation_30_clients  --baseline_accuracy 0.82225 --baseline_model training_data/dutch_30_clients/5_0.5_3848704714/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_30_clients/", num_clients=1, run_name=f"rc_{i}_individual",
#                                 project_name="DutchMultiplicityEvaluation_30_clients", dataset_name="dutch")

# # # Thresholds evaluation, (you only need 60% of the clients to be in the epsilon-ball)

# for t in [0.6, 0.75, 0.9]:
#     cmd = f"python  multiplicity_evaluate.py --num_clients 30 --sampled_test_nodes_per_round 30 --threshold {t} --dataset_name dutch --fed_dir training_data/dutch_30_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name {t}_global --project_name DutchMultiplicityEvaluation_30_clients  --baseline_accuracy 0.82225 --baseline_model training_data/dutch_30_clients/5_0.5_3848704714/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_30_clients/", num_clients=30, run_name=f"rc_{t}_global",
#                                 project_name="DutchMultiplicityEvaluation_30_clients", dataset_name="dutch")















# # #Evaluate the global rashomon set and then evaluates 10 times the individual rashomon sets

# cmd = f"python  multiplicity_evaluate.py --num_clients 40 --sampled_test_nodes_per_round 1.0 --dataset_name dutch --fed_dir training_data/dutch_40_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name global --project_name DutchMultiplicityEvaluation_40_clients  --baseline_accuracy 0.82175 --baseline_model training_data/dutch_40_clients/5_0.5_1288064256/"
# pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                         shell=True, preexec_fn=os.setsid)
# pro4.wait()
# calculate_rashomon_capacity(path="training_data/dutch_40_clients/", num_clients=40, run_name="rc_global", project_name="DutchMultiplicityEvaluation_40_clients", dataset_name="dutch")


# for i in range(10):
#     cmd = f"python  multiplicity_evaluate.py --num_clients 40 --sampled_test_nodes_per_round 0.05 --dataset_name dutch --fed_dir training_data/dutch_40_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name individual_{i} --project_name DutchMultiplicityEvaluation_40_clients  --baseline_accuracy 0.82175 --baseline_model training_data/dutch_40_clients/5_0.5_1288064256/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_40_clients/", num_clients=1, run_name=f"rc_{i}_individual",
#                                 project_name="DutchMultiplicityEvaluation_40_clients", dataset_name="dutch")

# # # Thresholds evaluation, (you only need 60% of the clients to be in the epsilon-ball)

# for t in [0.6, 0.75, 0.9]:
#     cmd = f"python  multiplicity_evaluate.py --num_clients 40 --sampled_test_nodes_per_round 40 --threshold {t} --dataset_name dutch --fed_dir training_data/dutch_40_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name {t}_global --project_name DutchMultiplicityEvaluation_40_clients  --baseline_accuracy 0.82175 --baseline_model training_data/dutch_40_clients/5_0.5_1288064256/"
#     pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                             shell=True, preexec_fn=os.setsid)
#     pro4.wait()
#     calculate_rashomon_capacity(path="training_data/dutch_40_clients/", num_clients=40, run_name=f"rc_{t}_global",
#                                 project_name="DutchMultiplicityEvaluation_40_clients", dataset_name="dutch")














# #Evaluate the global rashomon set and then evaluates 10 times the individual rashomon sets
cmd = "python  multiplicity_evaluate.py --num_clients 50 --sampled_test_nodes_per_round 1.0 --dataset_name dutch --fed_dir training_data/dutch_50_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name global --project_name DutchMultiplicityEvaluation_50_clients  --baseline_accuracy 0.81828 --baseline_model training_data/dutch_50_clients/5_1.0_4/"
pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        shell=True, preexec_fn=os.setsid)
pro4.wait()
calculate_rashomon_capacity(path="training_data/dutch_50_clients/", num_clients=50, run_name="rc_global", project_name="DutchMultiplicityEvaluation_50_clients", dataset_name="dutch")


for i in range(10):
    cmd = f"python  multiplicity_evaluate.py --num_clients 50 --sampled_test_nodes_per_round 0.05 --dataset_name dutch --fed_dir training_data/dutch_50_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name individual_{i} --project_name DutchMultiplicityEvaluation_50_clients  --baseline_accuracy 0.81828 --baseline_model training_data/dutch_50_clients/5_1.0_4/"
    pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            shell=True, preexec_fn=os.setsid)
    pro4.wait()
    calculate_rashomon_capacity(path="training_data/dutch_50_clients/", num_clients=1, run_name=f"rc_{i}_individual",
                                project_name="DutchMultiplicityEvaluation_50_clients", dataset_name="dutch")

# # Thresholds evaluation, (you only need 60% of the clients to be in the epsilon-ball)

for t in [0.6, 0.75, 0.9]:
    cmd = f"python  multiplicity_evaluate.py --num_clients 50 --sampled_test_nodes_per_round 50 --threshold {t} --dataset_name dutch --fed_dir training_data/dutch_50_clients/  --wandb True --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --run_name {t}_global --project_name DutchMultiplicityEvaluation_50_clients  --baseline_accuracy 0.81828 --baseline_model training_data/dutch_50_clients/5_1.0_4/"
    pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            shell=True, preexec_fn=os.setsid)
    pro4.wait()
    calculate_rashomon_capacity(path="training_data/dutch_50_clients/", num_clients=50, run_name=f"rc_{t}_global",
                                project_name="DutchMultiplicityEvaluation_50_clients", dataset_name="dutch")
