import subprocess
import os
import time

# for sample_clients in [0.25, 0.5, 0.75, 1.0]:
#     for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
#         for i in range(10):
#             cmd = f"python  main.py --FL_setting cross_silo --num_clients 20 --dataset_name mnist --num_rounds {num_rounds} --sampled_train_nodes_per_round {sample_clients} \
#             --sampled_test_nodes_per_round 0.25 --sampled_validation_nodes_per_round 0.0 --seed {i} --node_shuffle_seed 42 --batch_size=93 --lr 0.000565467849605206 --momentum 0.2035671836742751 --weight_decay 0.0004659291156319607 \
#              --optimizer adam --fed_dir training_data/mnist/ --project_name Mnist_multiplicity_20_clients --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
#               --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --num_epochs 2"
#             pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                                     shell=True, preexec_fn=os.setsid)
#             pro4.wait()



# for sample_clients in [0.25, 0.5, 0.75, 1.0]:
#     for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
#         for i in range(10):
#             cmd = f"python  main.py --FL_setting cross_silo --num_clients 50 --dataset_name mnist --num_rounds {num_rounds} --sampled_train_nodes_per_round {sample_clients} \
#             --sampled_test_nodes_per_round 0.25 --sampled_validation_nodes_per_round 0.0 --seed {i} --node_shuffle_seed 42 --batch_size=93 --lr 0.000565467849605206 --momentum 0.2035671836742751 --weight_decay 0.0004659291156319607 \
#              --optimizer adam --fed_dir training_data/mnist_50/ --project_name Mnist_multiplicity_50_clients --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
#               --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 1 --partitioner_by label --num_epochs 2"
#             pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
#                                     shell=True, preexec_fn=os.setsid)
#             pro4.wait()


for sample_clients in [0.25, 0.5, 0.75, 1.0]:
    for num_rounds in [10]:#, 25, 30, 35, 40, 45, 50, 55]:
        for i in range(10):
            cmd = f"python  main.py --FL_setting cross_silo --num_clients 20 --dataset_name mnist --num_rounds {num_rounds} --sampled_train_nodes_per_round {sample_clients} \
            --sampled_test_nodes_per_round 0.25 --sampled_validation_nodes_per_round 0.0 --seed {i} --node_shuffle_seed 42 --batch_size=93 --lr 0.000565467849605206 --momentum 0.2035671836742751 --weight_decay 0.0004659291156319607 \
             --optimizer adam --fed_dir training_data/mnist/ --project_name Mnist_multiplicity_20_clients_10 --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
              --dataset_path ./data/MNIST/train/ --partitioner_type non_iid --partitioner_alpha 10 --partitioner_by label --num_epochs 2"
            pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    shell=True, preexec_fn=os.setsid)
            pro4.wait()



