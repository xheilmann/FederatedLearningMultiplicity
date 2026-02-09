import subprocess
import os
import time

# for sample_clients in [0.25, 0.5, 0.75, 1.0]:
#     for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
#         for i in range(10):
#             cmd = f"python  main.py --FL_setting cross_silo --num_clients 20 --dataset_name dutch --num_rounds {num_rounds} --sampled_train_nodes_per_round {sample_clients} \
#             --sampled_test_nodes_per_round 0.25 --sampled_validation_nodes_per_round 0.0 --node_shuffle_seed 42 --batch_size=91 --lr 0.07679743675161778 --momentum 0.11217108296160291 --weight_decay 0.0006705640560089589 \
#              --optimizer adam --fed_dir training_data/dutch_20_clients/ --project_name Dutch_FL_multiplicity_20_clients_2011 --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
#               --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_epochs 3"
#             pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                                     shell=True, preexec_fn=os.setsid)
#             pro4.wait()




# for sample_clients in [0.25, 0.5, 0.75, 1.0]:
#     for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
#         for i in range(20):
#             cmd = f"python  main.py --FL_setting cross_silo --num_clients 10 --dataset_name dutch --num_rounds {num_rounds} --sampled_train_nodes_per_round \
#             {sample_clients} --sampled_test_nodes_per_round 0.25 --node_shuffle_seed 42 --batch_size=91 --lr 0.07679743675161778 --momentum 0.11217108296160291 --weight_decay 0.0006705640560089589 \
#              --optimizer adam --fed_dir training_data/dutch_10_clients/ --project_name DutchModelsFL_10 --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
#               --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_epochs 3"
#             pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                                     shell=True, preexec_fn=os.setsid)
#             pro4.wait()



for sample_clients in [0.25, 0.5, 0.75, 1.0]:
    for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
        for i in range(10):
            cmd = f"python  main.py --FL_setting cross_silo --num_clients 50 --dataset_name dutch --num_rounds {num_rounds} --sampled_train_nodes_per_round {sample_clients} \
            --sampled_test_nodes_per_round 0.25 --sampled_validation_nodes_per_round 0.0 --node_shuffle_seed 42 --batch_size=91 --lr 0.07679743675161778 --momentum 0.11217108296160291 --weight_decay 0.0006705640560089589 \
             --optimizer adam --fed_dir training_data/dutch_50_clients/ --project_name Dutch_FL_multiplicity_50_clients_2011 --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
              --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_epochs 3"
            pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    shell=True, preexec_fn=os.setsid)
            pro4.wait()




# for sample_clients in [0.25, 0.5, 0.75, 1.0]:
#     for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
#         for i in range(20):
#             cmd = f"python  main.py --FL_setting cross_silo --num_clients 30 --dataset_name dutch --num_rounds {num_rounds} --sampled_train_nodes_per_round \
#             {sample_clients} --sampled_test_nodes_per_round 0.25 --node_shuffle_seed 42 --batch_size=91 --lr 0.07679743675161778 --momentum 0.11217108296160291 --weight_decay 0.0006705640560089589 \
#              --optimizer adam --fed_dir training_data/dutch/ --project_name DutchModelsFL_30 --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
#               --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_epochs 3"
#             pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                                     shell=True, preexec_fn=os.setsid)
#             pro4.wait()

# for sample_clients in [0.25, 0.5, 0.75, 1.0]:
#     for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
#         for i in range(20):
#             cmd = f"python  main.py --FL_setting cross_silo --num_clients 40 --dataset_name dutch --num_rounds {num_rounds} --sampled_train_nodes_per_round \
#             {sample_clients} --sampled_test_nodes_per_round 0.25 --node_shuffle_seed 42 --batch_size=91 --lr 0.07679743675161778 --momentum 0.11217108296160291 --weight_decay 0.0006705640560089589 \
#              --optimizer adam --fed_dir training_data/dutch/ --project_name DutchModelsFL_40 --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
#               --dataset_path data/dutch/dutch.csv --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_epochs 3"
#             pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                                     shell=True, preexec_fn=os.setsid)
#             pro4.wait()