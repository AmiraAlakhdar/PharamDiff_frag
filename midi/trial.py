import torch
from midi.metrics.pgmg_match_eval import match_score
from midi.datasets.pharmacophore_utils import getPharamacophoreCoords


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
import numpy as np

# Example SMILES string
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

feat_arr = torch.tensor([1, 2]).long()
coord_arr = torch.tensor(  [[0.0, 0.0, 0.0],[0.0, 1.0, 0.0]]).float()


# Print results
print("Pharmacophore Positions:\n", feat_arr.shape)
print("Pharmacophore Types:\n", coord_arr.shape)

out = match_score(smiles, feat_arr, coord_arr)
print(out)