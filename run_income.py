import subprocess
import os
import time

for sample_clients in [0.25, 0.5, 0.75, 1.0]:
    for num_rounds in [1,2,3,4,5,6,7,8,9,10]:#, 25, 30, 35, 40, 45, 50, 55]:
        for i in range(10):
            cmd = f"python  main.py --FL_setting cross_silo --num_clients 20 --dataset_name income --num_rounds {num_rounds} --sampled_train_nodes_per_round \
            {sample_clients} --sampled_test_nodes_per_round 0.25 --sampled_validation_nodes_per_round 0.0 --seed {i} --node_shuffle_seed 42 --batch_size 38 --lr 0.0726338578312667 \
             --optimizer adam --momentum 0.5373415482935225  --fed_dir training_data/income/ --project_name Income_Models_multiplicity_2011 --run_name {num_rounds}_{sample_clients}_{i} --wandb True\
              --dataset_path data/income_reduced/ --partitioner_type non_iid --partitioner_alpha 1.0 --partitioner_by occupation --num_epochs 1 --weight_decay 0.000418605274818734 "
            pro4 = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    shell=True, preexec_fn=os.setsid)
            pro4.wait()
