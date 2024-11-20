Global seed set to 0
/home/amira/.conda/envs/midi/lib/python3.9/site-packages/torch_geometric/data/lightning/datamodule.py:49: UserWarning: The 'shuffle=True' option is ignored in 'QM9DataModule'. Remove it from the argument list to disable this warning
  warnings.warn(f"The 'shuffle={kwargs['shuffle']}' option is "
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/home/amira/.conda/envs/midi/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: UserWarning: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
  warning_cache.warn(
[rank: 0] Global seed set to 0
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
NUM NODES IN STATISTICS Counter({9: 81177, 8: 13158, 7: 2242, 6: 443, 5: 76, 4: 17, 3: 5, 2: 2})
NUM NODES IN STATISTICS Counter({9: 16562, 8: 2764, 7: 489, 6: 68, 5: 18, 4: 6, 3: 1})
NUM NODES IN STATISTICS Counter({9: 10792, 8: 1799, 7: 304, 6: 68, 5: 13, 4: 3})
Marginal distribution of the classes: nodes: tensor([0.7245, 0.1133, 0.1597, 0.0025]) -- edges: tensor([0.7261, 0.2386, 0.0271, 0.0082, 0.0000]) -- charges: tensor([0.0077, 0.9705, 0.0218])
[2024-07-18 18:34:22,975][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2024-07-18 18:34:22,975][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 1 nodes.
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

You are using a CUDA device ('NVIDIA GeForce RTX 3090 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]

  | Name                  | Type                  | Params
----------------------------------------------------------------
0 | train_loss            | TrainLoss             | 0     
1 | train_metrics         | TrainMolecularMetrics | 0     
2 | val_metrics           | MetricCollection      | 0     
3 | val_nll               | NLL                   | 0     
4 | val_sampling_metrics  | SamplingMetrics       | 0     
5 | test_metrics          | MetricCollection      | 0     
6 | test_nll              | NLL                   | 0     
7 | test_sampling_metrics | SamplingMetrics       | 0     
8 | model                 | GraphTransformer      | 20.3 M
----------------------------------------------------------------
20.3 M    Trainable params
0         Non-trainable params
20.3 M    Total params
81.397    Total estimated model params size (MB)
wandb: Currently logged in as: aalakhda (edm_10708). Use `wandb login --relogin` to force relogin
wandb: wandb version 0.17.4 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.15.4
wandb: Run data is saved locally in /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/wandb/run-20240718_183424-uc4ip9h7
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run noh_ada3
wandb: ⭐️ View project at https://wandb.ai/edm_10708/MolDiffusion_pharmacophoreqm9
wandb: 🚀 View run at https://wandb.ai/edm_10708/MolDiffusion_pharmacophoreqm9/runs/uc4ip9h7
Epoch 0: val/epoch_NLL: 1104428800.00 -- val/pos_mse: 1104423680.00 -- val/X_kl: 3133.06 -- val/E_kl: 5033.30 -- val/charges_kl: 2182.42 
Val loss: 1104428800.0000 	 Best val loss:  100000000.0000

Val epoch 0 ends
Starting epoch 0
Train epoch 0 ends
Epoch 0 finished: pos: 1.14 -- X: 0.60 -- charges: 0.13 -- E: 0.62 -- y: -1.00 -- 102.7s 
Starting epoch 1
Train epoch 1 ends
Epoch 1 finished: pos: 1.01 -- X: 0.39 -- charges: 0.10 -- E: 0.41 -- y: -1.00 -- 101.4s 
Starting epoch 2
Train epoch 2 ends
Epoch 2 finished: pos: 0.94 -- X: 0.33 -- charges: 0.08 -- E: 0.34 -- y: -1.00 -- 101.0s 
Starting epoch 3
Train epoch 3 ends
Epoch 3 finished: pos: 0.84 -- X: 0.31 -- charges: 0.08 -- E: 0.32 -- y: -1.00 -- 100.9s 
Starting epoch 4
Train epoch 4 ends
Epoch 4 finished: pos: 0.78 -- X: 0.29 -- charges: 0.07 -- E: 0.31 -- y: -1.00 -- 100.7s 
Starting epoch 5
Train epoch 5 ends
Epoch 5 finished: pos: 0.73 -- X: 0.29 -- charges: 0.07 -- E: 0.30 -- y: -1.00 -- 100.8s 
Starting epoch 6
Train epoch 6 ends
Epoch 6 finished: pos: 0.66 -- X: 0.28 -- charges: 0.07 -- E: 0.30 -- y: -1.00 -- 100.7s 
Starting epoch 7
Train epoch 7 ends
Epoch 7 finished: pos: 0.60 -- X: 0.27 -- charges: 0.06 -- E: 0.30 -- y: -1.00 -- 100.7s 
Starting epoch 8
Train epoch 8 ends
Epoch 8 finished: pos: 0.56 -- X: 0.27 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.2s 
Starting epoch 9
Epoch 9: val/epoch_NLL: 307581728.00 -- val/pos_mse: 307581248.00 -- val/X_kl: 4049.86 -- val/E_kl: 4455.85 -- val/charges_kl: 1096.28 
Val loss: 307581728.0000 	 Best val loss:  100000000.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch9/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 435.84 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 817, Kekulize 17, other 7,  -- No error 183
Validity over 1024 molecules: 17.87%
Number of connected components of 1024 molecules: mean:1.13 max:4.00
Connected components of 1024 molecules: 87.99
Uniqueness: 100.00% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 17.09%
rdkit valid pharmacophore match: 17.09%
rdkit pharmacophore match among valid: 31.15%
pgmg pharmacophore match: 16.76%
pgmg valid pharmacophore match: 25.50%
pgmg pharmacophore match percentage: 11.04%
pgmg valid pharmacophore match percentage: 20.77%
Sparsity level on local rank 0: 67 %
Too many edges, skipping angle distance computation.
Sampling metrics {'sampling/NumNodesW1': 0.015, 'sampling/AtomTypesTV': 0.116, 'sampling/EdgeTypesTV': 0.12, 'sampling/ChargeW1': 0.039, 'sampling/ValencyW1': 0.408, 'sampling/BondLengthsW1': 0.061, 'sampling/AnglesW1': 0.0}
Sampling metrics done.
Val epoch 9 ends
Train epoch 9 ends
Epoch 9 finished: pos: 0.52 -- X: 0.27 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 554.6s 
Starting epoch 10
Train epoch 10 ends
Epoch 10 finished: pos: 0.48 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 101.4s 
Starting epoch 11
Train epoch 11 ends
Epoch 11 finished: pos: 0.44 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 101.6s 
Starting epoch 12
Train epoch 12 ends
Epoch 12 finished: pos: 0.41 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.7s 
Starting epoch 13
Train epoch 13 ends
Epoch 13 finished: pos: 0.39 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.6s 
Starting epoch 14
Train epoch 14 ends
Epoch 14 finished: pos: 0.37 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.6s 
Starting epoch 15
Train epoch 15 ends
Epoch 15 finished: pos: 0.35 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.6s 
Starting epoch 16
Train epoch 16 ends
Epoch 16 finished: pos: 0.34 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.6s 
Starting epoch 17
Train epoch 17 ends
Epoch 17 finished: pos: 0.33 -- X: 0.26 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.5s 
Starting epoch 18
Train epoch 18 ends
Epoch 18 finished: pos: 0.32 -- X: 0.25 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 100.5s 
Starting epoch 19
Epoch 19: val/epoch_NLL: 67134728.00 -- val/pos_mse: 67134304.00 -- val/X_kl: 3541.04 -- val/E_kl: 4099.51 -- val/charges_kl: 911.67 
Val loss: 67134728.0000 	 Best val loss:  67134728.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch19/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 417.31 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 383, Kekulize 10, other 8,  -- No error 623
Validity over 1024 molecules: 60.84%
Number of connected components of 1024 molecules: mean:1.24 max:4.00
Connected components of 1024 molecules: 77.83
Uniqueness: 99.36% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 45.51%
rdkit valid pharmacophore match: 45.51%
rdkit pharmacophore match among valid: 51.20%
pgmg pharmacophore match: 44.32%
pgmg valid pharmacophore match: 47.99%
pgmg pharmacophore match percentage: 30.37%
pgmg valid pharmacophore match percentage: 34.03%
Sparsity level on local rank 0: 72 %
Sampling metrics {'sampling/NumNodesW1': 0.02, 'sampling/AtomTypesTV': 0.117, 'sampling/EdgeTypesTV': 0.02, 'sampling/ChargeW1': 0.018, 'sampling/ValencyW1': 0.209, 'sampling/BondLengthsW1': 0.314, 'sampling/AnglesW1': 12.819}
Sampling metrics done.
Val epoch 19 ends
Train epoch 19 ends
Epoch 19 finished: pos: 0.32 -- X: 0.25 -- charges: 0.06 -- E: 0.29 -- y: -1.00 -- 540.0s 
Starting epoch 20
Train epoch 20 ends
Epoch 20 finished: pos: 0.31 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 21
Train epoch 21 ends
Epoch 21 finished: pos: 0.30 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.9s 
Starting epoch 22
Train epoch 22 ends
Epoch 22 finished: pos: 0.30 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 103.7s 
Starting epoch 23
Train epoch 23 ends
Epoch 23 finished: pos: 0.30 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 24
Train epoch 24 ends
Epoch 24 finished: pos: 0.29 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 25
Train epoch 25 ends
Epoch 25 finished: pos: 0.29 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 26
Train epoch 26 ends
Epoch 26 finished: pos: 0.28 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 27
Train epoch 27 ends
Epoch 27 finished: pos: 0.28 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 28
Train epoch 28 ends
Epoch 28 finished: pos: 0.27 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 29
Epoch 29: val/epoch_NLL: 44259836.00 -- val/pos_mse: 44259416.00 -- val/X_kl: 3474.80 -- val/E_kl: 3958.19 -- val/charges_kl: 923.52 
Val loss: 44259836.0000 	 Best val loss:  44259836.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch29/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 421.25 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 308, Kekulize 5, other 8,  -- No error 703
Validity over 1024 molecules: 68.65%
Number of connected components of 1024 molecules: mean:1.23 max:3.00
Connected components of 1024 molecules: 79.30
Uniqueness: 99.86% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 46.48%
rdkit valid pharmacophore match: 46.48%
rdkit pharmacophore match among valid: 51.35%
pgmg pharmacophore match: 48.80%
pgmg valid pharmacophore match: 54.05%
pgmg pharmacophore match percentage: 36.04%
pgmg valid pharmacophore match percentage: 41.68%
Sparsity level on local rank 0: 73 %
Sampling metrics {'sampling/NumNodesW1': 0.015, 'sampling/AtomTypesTV': 0.144, 'sampling/EdgeTypesTV': 0.021, 'sampling/ChargeW1': 0.005, 'sampling/ValencyW1': 0.161, 'sampling/BondLengthsW1': 0.05, 'sampling/AnglesW1': 10.874}
Sampling metrics done.
Val epoch 29 ends
Train epoch 29 ends
Epoch 29 finished: pos: 0.27 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 543.8s 
Starting epoch 30
Train epoch 30 ends
Epoch 30 finished: pos: 0.27 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.3s 
Starting epoch 31
Train epoch 31 ends
Epoch 31 finished: pos: 0.27 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 32
Train epoch 32 ends
Epoch 32 finished: pos: 0.27 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 33
Train epoch 33 ends
Epoch 33 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 34
Train epoch 34 ends
Epoch 34 finished: pos: 0.26 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.1s 
Starting epoch 35
Train epoch 35 ends
Epoch 35 finished: pos: 0.26 -- X: 0.25 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 36
Train epoch 36 ends
Epoch 36 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.1s 
Starting epoch 37
Train epoch 37 ends
Epoch 37 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 38
Train epoch 38 ends
Epoch 38 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 39
Epoch 39: val/epoch_NLL: 17809644.00 -- val/pos_mse: 17809230.00 -- val/X_kl: 3415.30 -- val/E_kl: 3885.12 -- val/charges_kl: 894.20 
Val loss: 17809644.0000 	 Best val loss:  17809644.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch39/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 429.31 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 243, Kekulize 5, other 4,  -- No error 772
Validity over 1024 molecules: 75.39%
Number of connected components of 1024 molecules: mean:1.22 max:5.00
Connected components of 1024 molecules: 80.57
Uniqueness: 99.74% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 49.32%
rdkit valid pharmacophore match: 49.32%
rdkit pharmacophore match among valid: 52.85%
pgmg pharmacophore match: 50.96%
pgmg valid pharmacophore match: 54.73%
pgmg pharmacophore match percentage: 37.01%
pgmg valid pharmacophore match percentage: 41.58%
Sparsity level on local rank 0: 72 %
Sampling metrics {'sampling/NumNodesW1': 0.024, 'sampling/AtomTypesTV': 0.18, 'sampling/EdgeTypesTV': 0.019, 'sampling/ChargeW1': 0.016, 'sampling/ValencyW1': 0.105, 'sampling/BondLengthsW1': 0.034, 'sampling/AnglesW1': 11.241}
Sampling metrics done.
Val epoch 39 ends
Train epoch 39 ends
Epoch 39 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 555.5s 
Starting epoch 40
Train epoch 40 ends
Epoch 40 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.1s 
Starting epoch 41
Train epoch 41 ends
Epoch 41 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.1s 
Starting epoch 42
Train epoch 42 ends
Epoch 42 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 43
Train epoch 43 ends
Epoch 43 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 44
Train epoch 44 ends
Epoch 44 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.2s 
Starting epoch 45
Train epoch 45 ends
Epoch 45 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 46
Train epoch 46 ends
Epoch 46 finished: pos: 0.26 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 47
Train epoch 47 ends
Epoch 47 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.1s 
Starting epoch 48
Train epoch 48 ends
Epoch 48 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.5s 
Starting epoch 49
Epoch 49: val/epoch_NLL: 8481493.00 -- val/pos_mse: 8481093.00 -- val/X_kl: 3230.72 -- val/E_kl: 3860.63 -- val/charges_kl: 839.18 
Val loss: 8481493.0000 	 Best val loss:  8481493.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch49/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 433.38 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 168, Kekulize 3, other 3,  -- No error 850
Validity over 1024 molecules: 83.01%
Number of connected components of 1024 molecules: mean:1.28 max:4.00
Connected components of 1024 molecules: 75.78
Uniqueness: 99.18% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 57.91%
rdkit valid pharmacophore match: 57.91%
rdkit pharmacophore match among valid: 59.29%
pgmg pharmacophore match: 61.70%
pgmg valid pharmacophore match: 63.16%
pgmg pharmacophore match percentage: 46.97%
pgmg valid pharmacophore match percentage: 49.65%
Sparsity level on local rank 0: 73 %
Sampling metrics {'sampling/NumNodesW1': 0.006, 'sampling/AtomTypesTV': 0.075, 'sampling/EdgeTypesTV': 0.027, 'sampling/ChargeW1': 0.016, 'sampling/ValencyW1': 0.245, 'sampling/BondLengthsW1': 0.044, 'sampling/AnglesW1': 9.195}
Sampling metrics done.
Val epoch 49 ends
Train epoch 49 ends
Epoch 49 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 556.9s 
Starting epoch 50
Train epoch 50 ends
Epoch 50 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 51
Train epoch 51 ends
Epoch 51 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.8s 
Starting epoch 52
Train epoch 52 ends
Epoch 52 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 53
Train epoch 53 ends
Epoch 53 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 54
Train epoch 54 ends
Epoch 54 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 55
Train epoch 55 ends
Epoch 55 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 56
Train epoch 56 ends
Epoch 56 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 57
Train epoch 57 ends
Epoch 57 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 58
Train epoch 58 ends
Epoch 58 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 59
Epoch 59: val/epoch_NLL: 9144711.00 -- val/pos_mse: 9144312.00 -- val/X_kl: 3181.64 -- val/E_kl: 3867.91 -- val/charges_kl: 835.47 
Val loss: 9144711.0000 	 Best val loss:  8481493.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch59/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 419.73 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 186, Kekulize 7, other 3,  -- No error 828
Validity over 1024 molecules: 80.86%
Number of connected components of 1024 molecules: mean:1.15 max:3.00
Connected components of 1024 molecules: 85.64
Uniqueness: 99.52% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 56.45%
rdkit valid pharmacophore match: 56.45%
rdkit pharmacophore match among valid: 59.18%
pgmg pharmacophore match: 58.11%
pgmg valid pharmacophore match: 61.53%
pgmg pharmacophore match percentage: 45.21%
pgmg valid pharmacophore match percentage: 49.64%
Sparsity level on local rank 0: 73 %
Sampling metrics {'sampling/NumNodesW1': 0.014, 'sampling/AtomTypesTV': 0.12, 'sampling/EdgeTypesTV': 0.021, 'sampling/ChargeW1': 0.003, 'sampling/ValencyW1': 0.131, 'sampling/BondLengthsW1': 0.035, 'sampling/AnglesW1': 7.458}
Sampling metrics done.
Val epoch 59 ends
Train epoch 59 ends
Epoch 59 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 543.9s 
Starting epoch 60
Train epoch 60 ends
Epoch 60 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 61
Train epoch 61 ends
Epoch 61 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.8s 
Starting epoch 62
Train epoch 62 ends
Epoch 62 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.3s 
Starting epoch 63
Train epoch 63 ends
Epoch 63 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 103.6s 
Starting epoch 64
Train epoch 64 ends
Epoch 64 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.0s 
Starting epoch 65
Train epoch 65 ends
Epoch 65 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 66
Train epoch 66 ends
Epoch 66 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 67
Train epoch 67 ends
Epoch 67 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.1s 
Starting epoch 68
Train epoch 68 ends
Epoch 68 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 69
Epoch 69: val/epoch_NLL: 7800513.00 -- val/pos_mse: 7800105.50 -- val/X_kl: 3359.96 -- val/E_kl: 3895.22 -- val/charges_kl: 804.15 
Val loss: 7800513.0000 	 Best val loss:  7800513.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch69/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 418.93 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 230, Kekulize 2, other 0,  -- No error 792
Validity over 1024 molecules: 77.34%
Number of connected components of 1024 molecules: mean:1.17 max:4.00
Connected components of 1024 molecules: 84.38
Uniqueness: 100.00% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 49.90%
rdkit valid pharmacophore match: 49.90%
rdkit pharmacophore match among valid: 53.54%
pgmg pharmacophore match: 53.81%
pgmg valid pharmacophore match: 57.24%
pgmg pharmacophore match percentage: 40.92%
pgmg valid pharmacophore match percentage: 44.32%
Sparsity level on local rank 0: 73 %
Sampling metrics {'sampling/NumNodesW1': 0.023, 'sampling/AtomTypesTV': 0.213, 'sampling/EdgeTypesTV': 0.016, 'sampling/ChargeW1': 0.007, 'sampling/ValencyW1': 0.12, 'sampling/BondLengthsW1': 0.033, 'sampling/AnglesW1': 9.277}
Sampling metrics done.
Val epoch 69 ends
Train epoch 69 ends
Epoch 69 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 542.4s 
Starting epoch 70
Train epoch 70 ends
Epoch 70 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.4s 
Starting epoch 71
Train epoch 71 ends
Epoch 71 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 72
Train epoch 72 ends
Epoch 72 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 73
Train epoch 73 ends
Epoch 73 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 74
Train epoch 74 ends
Epoch 74 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 75
Train epoch 75 ends
Epoch 75 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 76
Train epoch 76 ends
Epoch 76 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 103.6s 
Starting epoch 77
Train epoch 77 ends
Epoch 77 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.5s 
Starting epoch 78
Train epoch 78 ends
Epoch 78 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 79
Epoch 79: val/epoch_NLL: 4650260.00 -- val/pos_mse: 4649866.00 -- val/X_kl: 3171.57 -- val/E_kl: 3753.49 -- val/charges_kl: 848.74 
Val loss: 4650260.0000 	 Best val loss:  4650260.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch79/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 424.74 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 165, Kekulize 0, other 2,  -- No error 857
Validity over 1024 molecules: 83.69%
Number of connected components of 1024 molecules: mean:1.30 max:4.00
Connected components of 1024 molecules: 74.61
Uniqueness: 99.42% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 51.27%
rdkit valid pharmacophore match: 51.27%
rdkit pharmacophore match among valid: 51.81%
pgmg pharmacophore match: 55.58%
pgmg valid pharmacophore match: 56.59%
pgmg pharmacophore match percentage: 41.70%
pgmg valid pharmacophore match percentage: 43.41%
Sparsity level on local rank 0: 74 %
Sampling metrics {'sampling/NumNodesW1': 0.02, 'sampling/AtomTypesTV': 0.198, 'sampling/EdgeTypesTV': 0.042, 'sampling/ChargeW1': 0.011, 'sampling/ValencyW1': 0.133, 'sampling/BondLengthsW1': 0.035, 'sampling/AnglesW1': 7.434}
Sampling metrics done.
Val epoch 79 ends
Train epoch 79 ends
Epoch 79 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 547.6s 
Starting epoch 80
Train epoch 80 ends
Epoch 80 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.3s 
Starting epoch 81
Train epoch 81 ends
Epoch 81 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.6s 
Starting epoch 82
Train epoch 82 ends
Epoch 82 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.3s 
Starting epoch 83
Train epoch 83 ends
Epoch 83 finished: pos: 0.24 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 84
Train epoch 84 ends
Epoch 84 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 85
Train epoch 85 ends
Epoch 85 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.3s 
Starting epoch 86
Train epoch 86 ends
Epoch 86 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.2s 
Starting epoch 87
Train epoch 87 ends
Epoch 87 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 88
Train epoch 88 ends
Epoch 88 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.8s 
Starting epoch 89
Epoch 89: val/epoch_NLL: 6199296.50 -- val/pos_mse: 6198902.50 -- val/X_kl: 3133.02 -- val/E_kl: 3803.26 -- val/charges_kl: 857.39 
Val loss: 6199296.5000 	 Best val loss:  4650260.0000

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch89/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 413.64 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 115, Kekulize 1, other 3,  -- No error 905
Validity over 1024 molecules: 88.38%
Number of connected components of 1024 molecules: mean:1.39 max:4.00
Connected components of 1024 molecules: 67.29
Uniqueness: 97.90% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 57.52%
rdkit valid pharmacophore match: 57.52%
rdkit pharmacophore match among valid: 55.47%
pgmg pharmacophore match: 60.89%
pgmg valid pharmacophore match: 59.69%
pgmg pharmacophore match percentage: 46.78%
pgmg valid pharmacophore match percentage: 46.08%
Sparsity level on local rank 0: 76 %
Sampling metrics {'sampling/NumNodesW1': 0.029, 'sampling/AtomTypesTV': 0.139, 'sampling/EdgeTypesTV': 0.072, 'sampling/ChargeW1': 0.005, 'sampling/ValencyW1': 0.296, 'sampling/BondLengthsW1': 0.035, 'sampling/AnglesW1': 5.548}
Sampling metrics done.
Val epoch 89 ends
Train epoch 89 ends
Epoch 89 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 538.4s 
Starting epoch 90
Train epoch 90 ends
Epoch 90 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.4s 
Starting epoch 91
Train epoch 91 ends
Epoch 91 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 104.0s 
Starting epoch 92
Train epoch 92 ends
Epoch 92 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.4s 
Starting epoch 93
Train epoch 93 ends
Epoch 93 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 94
Train epoch 94 ends
Epoch 94 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 95
Train epoch 95 ends
Epoch 95 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 96
Train epoch 96 ends
Epoch 96 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 97
Train epoch 97 ends
Epoch 97 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 101.1s 
Starting epoch 98
Train epoch 98 ends
Epoch 98 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.5s 
Starting epoch 99
Epoch 99: val/epoch_NLL: 3394701.75 -- val/pos_mse: 3394304.75 -- val/X_kl: 3133.62 -- val/E_kl: 3825.82 -- val/charges_kl: 873.44 
Val loss: 3394701.7500 	 Best val loss:  3394701.7500

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch99/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 423.81 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 191, Kekulize 0, other 1,  -- No error 832
Validity over 1024 molecules: 81.25%
Number of connected components of 1024 molecules: mean:1.15 max:3.00
Connected components of 1024 molecules: 85.84
Uniqueness: 99.76% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 54.69%
rdkit valid pharmacophore match: 54.69%
rdkit pharmacophore match among valid: 57.81%
pgmg pharmacophore match: 59.55%
pgmg valid pharmacophore match: 62.92%
pgmg pharmacophore match percentage: 46.58%
pgmg valid pharmacophore match percentage: 50.60%
Sparsity level on local rank 0: 73 %
Sampling metrics {'sampling/NumNodesW1': 0.019, 'sampling/AtomTypesTV': 0.179, 'sampling/EdgeTypesTV': 0.011, 'sampling/ChargeW1': 0.01, 'sampling/ValencyW1': 0.106, 'sampling/BondLengthsW1': 0.043, 'sampling/AnglesW1': 7.774}
Sampling metrics done.
Val epoch 99 ends
Train epoch 99 ends
Epoch 99 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 548.2s 
Starting epoch 100
Train epoch 100 ends
Epoch 100 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 101
Train epoch 101 ends
Epoch 101 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 102
Train epoch 102 ends
Epoch 102 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 103.6s 
Starting epoch 103
Train epoch 103 ends
Epoch 103 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.27 -- y: -1.00 -- 100.6s 
Starting epoch 104
Train epoch 104 ends
Epoch 104 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 105
Train epoch 105 ends
Epoch 105 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 106
Train epoch 106 ends
Epoch 106 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 107
Train epoch 107 ends
Epoch 107 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 108
Train epoch 108 ends
Epoch 108 finished: pos: 0.25 -- X: 0.24 -- charges: 0.06 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 109
Epoch 109: val/epoch_NLL: 5329441.50 -- val/pos_mse: 5329054.50 -- val/X_kl: 3120.71 -- val/E_kl: 3727.35 -- val/charges_kl: 803.33 
Val loss: 5329441.5000 	 Best val loss:  3394701.7500

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch109/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 420.93 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 131, Kekulize 1, other 0,  -- No error 892
Validity over 1024 molecules: 87.11%
Number of connected components of 1024 molecules: mean:1.40 max:4.00
Connected components of 1024 molecules: 67.77
Uniqueness: 98.09% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 54.30%
rdkit valid pharmacophore match: 54.30%
rdkit pharmacophore match among valid: 53.48%
pgmg pharmacophore match: 58.22%
pgmg valid pharmacophore match: 57.64%
pgmg pharmacophore match percentage: 44.92%
pgmg valid pharmacophore match percentage: 45.29%
Sparsity level on local rank 0: 75 %
Sampling metrics {'sampling/NumNodesW1': 0.018, 'sampling/AtomTypesTV': 0.132, 'sampling/EdgeTypesTV': 0.065, 'sampling/ChargeW1': 0.006, 'sampling/ValencyW1': 0.281, 'sampling/BondLengthsW1': 0.038, 'sampling/AnglesW1': 5.019}
Sampling metrics done.
Val epoch 109 ends
Train epoch 109 ends
Epoch 109 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 543.2s 
Starting epoch 110
Train epoch 110 ends
Epoch 110 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.2s 
Starting epoch 111
Train epoch 111 ends
Epoch 111 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.8s 
Starting epoch 112
Train epoch 112 ends
Epoch 112 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 113
Train epoch 113 ends
Epoch 113 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.27 -- y: -1.00 -- 100.6s 
Starting epoch 114
Train epoch 114 ends
Epoch 114 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 115
Train epoch 115 ends
Epoch 115 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.1s 
Starting epoch 116
Train epoch 116 ends
Epoch 116 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 117
Train epoch 117 ends
Epoch 117 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 118
Train epoch 118 ends
Epoch 118 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 103.0s 
Starting epoch 119
Epoch 119: val/epoch_NLL: 1894858.38 -- val/pos_mse: 1894470.38 -- val/X_kl: 3120.08 -- val/E_kl: 3746.04 -- val/charges_kl: 795.11 
Val loss: 1894858.3750 	 Best val loss:  1894858.3750

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch119/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 420.77 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 153, Kekulize 0, other 1,  -- No error 870
Validity over 1024 molecules: 84.96%
Number of connected components of 1024 molecules: mean:1.18 max:3.00
Connected components of 1024 molecules: 83.69
Uniqueness: 99.66% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 54.59%
rdkit valid pharmacophore match: 54.59%
rdkit pharmacophore match among valid: 57.13%
pgmg pharmacophore match: 57.75%
pgmg valid pharmacophore match: 60.19%
pgmg pharmacophore match percentage: 45.70%
pgmg valid pharmacophore match percentage: 48.39%
Sparsity level on local rank 0: 73 %
Sampling metrics {'sampling/NumNodesW1': 0.011, 'sampling/AtomTypesTV': 0.143, 'sampling/EdgeTypesTV': 0.022, 'sampling/ChargeW1': 0.012, 'sampling/ValencyW1': 0.158, 'sampling/BondLengthsW1': 0.054, 'sampling/AnglesW1': 7.086}
Sampling metrics done.
Val epoch 119 ends
Train epoch 119 ends
Epoch 119 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 544.3s 
Starting epoch 120
Train epoch 120 ends
Epoch 120 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.3s 
Starting epoch 121
Train epoch 121 ends
Epoch 121 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.8s 
Starting epoch 122
Train epoch 122 ends
Epoch 122 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.7s 
Starting epoch 123
Train epoch 123 ends
Epoch 123 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.27 -- y: -1.00 -- 100.5s 
Starting epoch 124
Train epoch 124 ends
Epoch 124 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.5s 
Starting epoch 125
Train epoch 125 ends
Epoch 125 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.27 -- y: -1.00 -- 100.5s 
Starting epoch 126
Train epoch 126 ends
Epoch 126 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.5s 
Starting epoch 127
Train epoch 127 ends
Epoch 127 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 128
Train epoch 128 ends
Epoch 128 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 129
Epoch 129: val/epoch_NLL: 4242876.00 -- val/pos_mse: 4242478.50 -- val/X_kl: 3205.07 -- val/E_kl: 3814.89 -- val/charges_kl: 823.56 
Val loss: 4242876.0000 	 Best val loss:  1894858.3750

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch129/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 409.32 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 187, Kekulize 0, other 0,  -- No error 837
Validity over 1024 molecules: 81.74%
Number of connected components of 1024 molecules: mean:1.21 max:4.00
Connected components of 1024 molecules: 81.74
Uniqueness: 99.64% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 54.59%
rdkit valid pharmacophore match: 54.59%
rdkit pharmacophore match among valid: 56.39%
pgmg pharmacophore match: 59.83%
pgmg valid pharmacophore match: 62.94%
pgmg pharmacophore match percentage: 46.39%
pgmg valid pharmacophore match percentage: 49.22%
Sparsity level on local rank 0: 74 %
Sampling metrics {'sampling/NumNodesW1': 0.007, 'sampling/AtomTypesTV': 0.182, 'sampling/EdgeTypesTV': 0.036, 'sampling/ChargeW1': 0.006, 'sampling/ValencyW1': 0.104, 'sampling/BondLengthsW1': 0.029, 'sampling/AnglesW1': 5.898}
Sampling metrics done.
Val epoch 129 ends
Train epoch 129 ends
Epoch 129 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 532.5s 
Starting epoch 130
Train epoch 130 ends
Epoch 130 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.1s 
Starting epoch 131
Train epoch 131 ends
Epoch 131 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.5s 
Starting epoch 132
Train epoch 132 ends
Epoch 132 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 101.1s 
Starting epoch 133
Train epoch 133 ends
Epoch 133 finished: pos: 0.24 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.6s 
Starting epoch 134
Train epoch 134 ends
Epoch 134 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.5s 
Starting epoch 135
Train epoch 135 ends
Epoch 135 finished: pos: 0.25 -- X: 0.24 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.9s 
Starting epoch 136
Train epoch 136 ends
Epoch 136 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 100.8s 
Starting epoch 137
Train epoch 137 ends
Epoch 137 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.28 -- y: -1.00 -- 103.2s 
Starting epoch 138
Train epoch 138 ends
Epoch 138 finished: pos: 0.24 -- X: 0.23 -- charges: 0.05 -- E: 0.27 -- y: -1.00 -- 100.3s 
Starting epoch 139
Epoch 139: val/epoch_NLL: 3478974.75 -- val/pos_mse: 3478592.25 -- val/X_kl: 3022.32 -- val/E_kl: 3723.64 -- val/charges_kl: 821.23 
Val loss: 3478974.7500 	 Best val loss:  1894858.3750

Sampling start
Sampling a batch with 20 graphs. Saving 20 visualization and 1 full chains.
Batch sampled. Visualizing chains starts!
Visualizing chain 0/1
Molecule list generated.
Saving the gif at /home/amira/midi_two/MiDi/outputs/2024-07-18/18-33-18-noh_ada3/chains/epoch139/batch20_GR0_0.gif.
Chain saved.
Visualizing 20 individual molecules...
Visualizing done.
Sampling a batch with 1004 graphs. Saving 0 visualization and 0 full chains.
Visualizing done.
Done on 0. Sampling took 428.19 seconds

Computing sampling metrics on 0...
Error messages: AtomValence 149, Kekulize 0, other 1,  -- No error 874
Validity over 1024 molecules: 85.35%
Number of connected components of 1024 molecules: mean:1.24 max:4.00
Connected components of 1024 molecules: 78.42
Uniqueness: 99.77% WARNING: do not trust this metric on multi-gpu
Novelty: 100.00%
rdkit pharmacophore match: 52.05%
rdkit valid pharmacophore match: 52.05%
rdkit pharmacophore match among valid: 54.69%
pgmg pharmacophore match: 56.85%
pgmg valid pharmacophore match: 59.76%
pgmg pharmacophore match percentage: 43.55%
pgmg valid pharmacophore match percentage: 47.37%
Sparsity level on local rank 0: 74 %
Sampling metrics {'sampling/NumNodesW1': 0.026, 'sampling/AtomTypesTV': 0.129, 'sampling/EdgeTypesTV': 0.034, 'sampling/ChargeW1': 0.014, 'sampling/ValencyW1': 0.177, 'sampling/BondLengthsW1': 0.029, 'sampling/AnglesW1': 5.828}
Sampling metrics done.
Val epoch 139 ends
Train epoch 139 ends
