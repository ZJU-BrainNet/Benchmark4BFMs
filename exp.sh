#!/bin/bash

# dataset
# 6 subjects MAYO FNUSA 
# 5 subjects UCSD_ON HUSM RepOD Siena SleepEDFx AD-65
# 4 subjects CHBMIT ISRUC

# model
# LaBraM Mbrain NeuroGPT NeuroLM SppEEGNet CBraMod EEGPT BrainBERT BIOT Bendr BFM Brant1
# for model in Brant1

# mode
# finetune test


python pretrained_run.py --run_mode finetune --gpu_id 4 --is_parallel False --model LaBraM --dataset MAYO --cv_id 0 --batch_size 128
