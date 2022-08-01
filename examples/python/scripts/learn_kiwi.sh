#!/bin/bash

python3 ../learn_real_data.py \
        --stats_folder "joint_kiwi" \
        --light_file "gantry" \
        --seed 2 \
        --scene "kiwi" \
        --n_dump 50 \
        --ref_folder "exr_ref" \
        --mesh_lr 0.05 \
        --sigma_lr 0.05 \
        --albedo_lr 0.05 \
        --rough_lr 0.05 \
        --eta_lr 0.005 \
        --n_reduce_step 1000 \
        --n_iters 10000 \
        --laplacian 30 \
        --spp 32 \
        --sppe 32 \
        --sppse 2500 \
        --sigma_laplacian 0 \
        --albedo_laplacian 0 \
        --rough_laplacian 0 \
        --albedo_texture 512 \
        --sigma_texture 0 \
        --rough_texture 512 \
        --no_init "no" \
        --d_type "real" 