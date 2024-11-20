# Do not move these imports, the order seems to matter
import torch
import pytorch_lightning as pl

import os
import warnings
import pathlib

import random

import hydra
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from midi.datasets import qm9_dataset, geom_dataset
from midi.metrics.molecular_metrics import SamplingMetrics
from torch.utils.data import Subset
from torch_geometric.loader.dataloader import DataLoader
from midi import utils
from torch_geometric.data import Batch, Dataset
from midi.analysis.rdkit_functions import Molecule


#from midi.diffusion_model import FullDenoisingDiffusion


def collate_fn(data):
    size = len(data)
    print(data[0])
    print(size)
    ligands = data.ligand 
    pharmacophores = data.pharmacophore
    
    phar_random_indices = random.sample(range(size), 1024)
    lignad_random_indices = random.sample(range(size), 1024)
    


    #ligands, pharmacophores = shuffle_data(ligands, pharmacophores )


    ligand_batch = Batch.from_data_list(ligands[lignad_random_indices])
    pharmacophore_batch = Batch.from_data_list(pharmacophores[phar_random_indices])
    #assert len(ligand_batch) == len(pharmacophore_batch)
    print('batch')
    print(len(pharmacophore_batch))
    print(len(ligand_batch))

    # Prepare the collated data
    data = {'ligand': ligand_batch,
            'pharmacophore': pharmacophore_batch}

    return data

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)

    if dataset_config.name in ['qm9', "geom"]:
        if dataset_config.name == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

        else:
            datamodule = geom_dataset.GeomDataModule(cfg)
            dataset_infos = geom_dataset.GeomInfos(datamodule=datamodule, cfg=cfg)
            
        train_data = datamodule.train_dataloader().dataset
        test_data = datamodule.test_dataloader().dataset
        val_data = datamodule.val_dataloader().dataset

        train_smiles = list(datamodule.train_dlenataloader().dataset.smiles) if cfg.general.test_only else []
        
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    # utils.create_folders(cfg)
    
    total_train_size = len(train_data)
    
    
    #train_data['ligand'] = train_data['ligand'][random_indices]
        
    # Randomly pick 1024 indices from the dataset
    random_indices = random.sample(range(total_train_size), 1024)
        
        # Create a subset of the dataset using the selected random indices
    subset_train_dataset = Subset(train_data, random_indices)
    
    #print(subset_train_dataset)
    
    subset_train_loader = DataLoader(subset_train_dataset, batch_size=1024, shuffle=False)
        
        # Iterate over the DataLoader to access the subset data
    
    random_sampling_metrics = SamplingMetrics(train_smiles, dataset_infos, test=False)

    print("Loading subset of training data...")
    
    for batch in subset_train_loader:
            # Process the batch as needed
        print("Batch:", batch)
        batch_size = 1024
    
        dense_batch = utils.to_dense(batch, dataset_infos)
        random_indices = torch.randperm(batch_size).tolist()
        
        dense_batch.pharma_coord = dense_batch.pharma_coord[random_indices]
        dense_batch.pharma_feat = dense_batch.pharma_feat[random_indices]
        dense_batch.pharma_mask = dense_batch.pharma_mask[random_indices]
        
        sampled = dense_batch.collapse(dataset_infos.collapse_charges)
        
        
        X, charges, E, y, pos = sampled.X, sampled.charges, sampled.E, sampled.y, sampled.pos
        pharma_coords, pharma_feats = sampled.pharma_coord, sampled.pharma_feat
        node_mask = sampled.node_mask
        n_nodes = node_mask.sum(dim=1)
        
        molecule_list = []
        print(batch_size)
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n]
            charge_vec = charges[i, :n]
            edge_types = E[i, :n, :n]
            conformer = pos[i, :n]
            molecule_list.append(Molecule(atom_types=atom_types, charges=charge_vec,
                                          bond_types=edge_types, positions=conformer,
                                          atom_decoder=dataset_infos.atom_decoder, 
                                          pharma_feat=pharma_feats[i], pharma_coord=pharma_coords[i]))

        
        random_sampling_metrics(molecule_list, "debug", 0, 0)    
        

if __name__ == '__main__':
    main()
