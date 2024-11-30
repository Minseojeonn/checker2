#!/bin/bash

#SBATCH -J 500
#SBATCH --gpus=1

for input_dim in 32 64
do
    for lr in 0.01 0.001
    do
        for layers in 1 3
        do
            for sign in 3 4
            do
                srun --gpu-bind=single:1 python -m main --sign ${sign} --input_dim ${input_dim} --lr ${lr} --num_layers ${layers} --seed 500
            done
        done
    done
done

exit 0
