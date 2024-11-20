import os.path
from collections import defaultdict
from typing import Iterable, Dict, List, Tuple
import random

import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem, Recap, Descriptors, SDWriter

import torch
from torch_geometric.data import Data
import torch.nn.functional as F

from rdkit.Chem import rdMMPA




PHARMACOPHORE_FAMILES_TO_KEEP= ('Aromatic', 'Hydrophobe', 'PosIonizable', 'Acceptor', 'Donor', 
                                 'LumpedHydrophobe')
FAMILY_MAPPING = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}

phar2idx = {"AROM": 1, "HYBL": 2, "POSC": 3, "HACC": 4, "HDON": 5, "LHYBL": 6, "UNKONWN": 0}
_FEATURES_FACTORY = []


def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = int(random.random() * n)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])

    return Psample



def classify_fragment(mol,  features_names: Iterable[str] = PHARMACOPHORE_FAMILES_TO_KEEP, confId:int=-1 ):
    classes = []
    coord_arr = np.empty(0)
    if mol is None:
        return None
    
    feature_factory, keep_featnames = get_features_factory(features_names)
    Chem.GetSSSR(mol)
    rawFeats = feature_factory.GetFeaturesForMol(mol, confId=confId)
    for f in rawFeats:
        if f.GetFamily() in features_names:
            classes.append(f.GetFamily())
            coord_arr= np.append(coord_arr,np.array(f.GetPos(confId=f.GetActiveConformer())))
    
    coord_arr = coord_arr.reshape(-1, 3)

    return classes, coord_arr




def get_features_factory(features_names, resetPharmacophoreFactory=False):
    global _FEATURES_FACTORY, _FEATURES_NAMES
    if resetPharmacophoreFactory or (len(_FEATURES_FACTORY) > 0 and _FEATURES_FACTORY[-1] != features_names):
        _FEATURES_FACTORY.pop()
        _FEATURES_FACTORY.pop()
    if len(_FEATURES_FACTORY) == 0:
        feature_factory = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        _FEATURES_NAMES = features_names
        if features_names is None:
            features_names = list(feature_factory.GetFeatureFamilies())

        _FEATURES_FACTORY.extend([feature_factory, features_names])
    return _FEATURES_FACTORY


# def getPharamacophoreCoords(mol, features_names: Iterable[str] = PHARMACOPHORE_FAMILES_TO_KEEP, confId:int=-1) -> \
#         Tuple[Dict[str, np.ndarray],  Dict[str, List[np.ndarray]]] :

#     feature_factory, keep_featnames = get_features_factory(features_names)
#     Chem.GetSSSR(mol)
#     rawFeats = feature_factory.GetFeaturesForMol(mol, confId=confId)

#     feat_arr = np.empty(0)
#     idx_arr_list = []
#     coord_arr = np.empty(0)


#     hydrophobes = set()  # To store atom indices of hydrophobes
#     lumped_hydrophobes_and_arom = set()  # To store atom indices of lumped hydrophobes
    

    
#     for f in rawFeats:
#         if f.GetFamily() in keep_featnames:
#             idx_arr = np.empty(0)
#             if len(f.GetAtomIds()) > 1:
#                 for idx in f.GetAtomIds():
#                     idx_arr = np.append(idx_arr, np.array(idx))

#             else:
#                 idx_arr = np.append(idx_arr, np.array(random.choice(list(f.GetAtomIds()))))
                
#             feat_arr = np.append(feat_arr, np.array(FAMILY_MAPPING[f.GetFamily()]))
#             coord_arr= np.append(coord_arr, np.array(f.GetPos(confId=f.GetActiveConformer())))
#             idx_arr_list.append(idx_arr)

        
#     coord_arr = coord_arr.reshape(-1, 3)
    

    
#     permuted_indices = np.random.permutation(range(len(feat_arr))).astype(int)
#     feat_arr = feat_arr[permuted_indices] 
#     idx_arr_list = [idx_arr_list[i] for i in permuted_indices]
#     coord_arr = coord_arr[permuted_indices]
    
    
#     new_feat_arr = np.empty(0)
#     new_idx_arr_list = []
#     new_coord_arr =  np.empty((0, coord_arr.shape[1]))
    
    
#     used_atoms = set()
    
#     for i, idx_list in enumerate(idx_arr_list):
#         if any(idx in idx_list for idx in used_atoms):
#             continue    
#         new_feat_arr = np.append(new_feat_arr, feat_arr[i])
#         new_coord_arr = np.append(new_coord_arr, [coord_arr[i]], axis=0) 
#         new_idx_arr_list.append(idx_list)
#         used_atoms.update(idx_list)

#     if new_feat_arr.shape[0] == 0:
#         return new_feat_arr, new_idx_arr_list, new_coord_arr
    
#     assert len(new_feat_arr) == len(new_idx_arr_list) == len(new_coord_arr), \
#     f"Length mismatch: feat_arr({len(new_feat_arr)}), idx_arr_list({len(new_idx_arr_list)}), coord_arr({len(new_coord_arr)})"

            
#     return new_feat_arr, new_idx_arr_list, new_coord_arr



# def pharmacophore_to_torch(feat_arr, idx_arr_list, coord_arr, pos_mean, mol, name):
    
#     n = mol.GetNumAtoms()
    
#     if len(feat_arr) < 2 :
#         return None
    
#     if name == 'qm9':
#         num = [2, 3, 4]
#         num_p = [0.333, 0.334, 0.333]  # P(Number of Pharmacophore points)
#         num_ = sample_probability(num, num_p, 1)
#     elif name == 'geom':
#         num = [3, 4, 5, 6, 7]
#         num_p = [0.086, 0.0864, 0.389, 0.495, 0.0273]  # P(Number of Pharmacophore points)
#         num_ = sample_probability(num, num_p, 1)    
    
#     if len(feat_arr)  >= int(num_[0]):
#         indices = np.random.choice(len(feat_arr), size=int(num_[0]), replace=False)
#         feat_arr = feat_arr[indices]
#         idx_arr_list = [idx_arr_list[i] for i in indices]
#         coord_arr = coord_arr[indices]
         
    
#     feat_array = np.zeros(n)
#     coord_array = np.zeros((n, 3))
#     mask_array = np.zeros(n)
    
    
#     for i, idx_list in enumerate(idx_arr_list):
#         for idx in idx_list:
#             feat_array[int(idx)] = feat_arr[i]
#             coord_array[int(idx)] = coord_arr[i]
#             mask_array[int(idx)] = 1
            
#     num_heavy_atoms = mol.GetNumHeavyAtoms()
    
#     # Check if the sum of the mask array exceeds the number of heavy atoms
#     if mask_array.sum() >= num_heavy_atoms:
#         print(f"Pharmacophore points ({mask_array.sum()}) are more than or equal to the number of heavy atoms in the molecule ({num_heavy_atoms}).")
#         return None

#     coord_array = torch.Tensor(coord_array).float()

#     coord_array = coord_array - pos_mean
    
#     pharma_feat = torch.Tensor(feat_array).long()
#     mask_array = torch.Tensor(mask_array).long()
#     coord_array = coord_array  * mask_array.unsqueeze(-1)
    
    
#     pharma = Data(x = pharma_feat, pos = coord_array, y=mask_array)

#     return pharma


# def mol_to_torch_pharmacophore(mol, pos_mean, name=None):
#     feat_arr, idx_arr, coord_arr = getPharamacophoreCoords(mol)
#     pharma_data =  pharmacophore_to_torch(feat_arr, idx_arr, coord_arr, pos_mean, mol, name=name)
#     return pharma_data


def mol_to_fragments(mol):
    
    fragments_mols_3d = []
    fragment_types = []
    atom_maps = []
    coords_list = []
    conf = mol.GetConformer()
    # Fragment the molecule using RECAP
    
    
    #Chem.GetSSSR(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    o = fragment_mol(smiles, smiles, design_task="linker")
    if o is None:
        return None, None, None, None
    for l in o:
        smis = l.replace('.',',').split(',')
        frags = [Chem.MolFromSmiles(smi) for smi in smis[2:] if smi != '']
    if len(frags) == 0:
        frags = [mol]  # Use the original molecule if no decompositi
    
    for frag in frags:
        
        # Editable molecule
        editable_frag = Chem.EditableMol(frag)

        # Find indices of dummy atoms (atomic number 0)
        dummy_indices = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]

        # Remove dummy atoms
        for idx in sorted(dummy_indices, reverse=True):  # Remove in reverse order to maintain indexing
            editable_frag.RemoveAtom(idx)
        
        frag = editable_frag.GetMol()
                
        # match = mol.GetSubstructMatch(frag)
        # if match:
        #     print("Match found:", match)
        # else:
        #     print("No match found.")
        
        atom_map = mol.GetSubstructMatch(frag)
            
        fragment_conf = Chem.Conformer(len(frag.GetAtoms()))
        for i, atom_idx in enumerate(atom_map):
            fragment_conf.SetAtomPosition(i, conf.GetAtomPosition(atom_idx))
            
        frag.AddConformer(fragment_conf, assignId=True)
        
        classification, coords = classify_fragment(frag)
        
      
        if classification and frag is not None:
            fragments_mols_3d.append(frag)  # Add fragment
            fragment_types.append(classification)    # Add corresponding type
            atom_maps.append(atom_map)
            coords_list.append(coords)
            
    return fragments_mols_3d, fragment_types, atom_maps, coords_list


def pharmacophore_frag_to_torch(fragments_mols_3d, fragment_types, atom_maps, coords_list, mol, name):
    
    n = mol.GetNumAtoms()
    
    if fragments_mols_3d is None:
        return None
    
    if len(fragments_mols_3d) < 2 :
        return None
    
    if name == 'qm9':
        num = [2, 3, 4]
        num_p = [0.333, 0.334, 0.333]  # P(Number of Pharmacophore points)
        num_ = sample_probability(num, num_p, 1)
    elif name == 'geom':
        num = [3, 4, 5, 6, 7]
        num_p = [0.086, 0.0864, 0.389, 0.495, 0.0273]  # P(Number of Pharmacophore points)
        num_ = sample_probability(num, num_p, 1)
        
    if len(fragments_mols_3d)  >= int(num_[0]):
        indices = random.sample(range(len(fragments_mols_3d)), num_[0])
    else:
        indices = list(range(len(fragments_mols_3d)))
            
    feat_array = np.zeros(n)
    mask_array = np.zeros(n)
    coord_array = np.zeros((n, 3))
    
    
        
    for idx in indices:
        #frag = fragments_mols_3d[idx]
        p_type_idx =  random.sample(range(len(fragment_types[idx])), 1)
        p_type = fragment_types[idx][p_type_idx[0]]
        atom_map = atom_maps[idx]
        p_coords = coords_list[idx][p_type_idx[0]]
            
        for atom in atom_map:
            feat_array[atom] = FAMILY_MAPPING[p_type]
            mask_array[atom] = 1
            coord_array[atom] = p_coords
            
    

    num_heavy_atoms = mol.GetNumHeavyAtoms()
    
    # Check if the sum of the mask array exceeds the number of heavy atoms
    if mask_array.sum() >= num_heavy_atoms:
        print(f"Pharmacophore points ({mask_array.sum()}) are more than or equal to the number of heavy atoms in the molecule ({num_heavy_atoms}).")
        return None

    coord_array = torch.Tensor(coord_array).float()
    
    pharma_feat = torch.Tensor(feat_array).long()
    mask_array = torch.Tensor(mask_array).long()
    #coord_array = coord_array - torch.mean(coord_array, dim=0, keepdim=True)
    coord_array = coord_array  * mask_array.unsqueeze(-1)
    
    pharma = Data(x = pharma_feat, pos = coord_array, y=mask_array)

    return pharma
            
                
def get_frag_dict(fragments_mols_3d, fragment_types, all_classes=PHARMACOPHORE_FAMILES_TO_KEEP):
    
    if fragments_mols_3d is None:
        return None
    
    classes_dict = {cls: [] for cls in all_classes}
    
    for fragment, frag_classes in zip(fragments_mols_3d, fragment_types):
        for cls in frag_classes:
            classes_dict[cls].append(fragment)
            
    return classes_dict

    
            
    
def mol_to_frag_pharmacophore(mol, name):
    fragments_mols_3d, fragment_types, atom_maps,  coords_list = mol_to_fragments(mol)
    pharma_data = pharmacophore_frag_to_torch(fragments_mols_3d, fragment_types, atom_maps, coords_list, mol, name)
    frag_dict = get_frag_dict(fragments_mols_3d, fragment_types)
    return pharma_data, frag_dict        
    


def load_phar_file(file_path):
    load_file_fn = {'.posp': load_pp_file}.get(file_path.suffix, None)

    if load_file_fn is None:
        raise ValueError(f'Invalid file path: "{file_path}"!')

    return load_file_fn(file_path)



def fragment_mol(smi, cid, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]", design_task="linker"):
    mol = Chem.MolFromSmiles(smi)

    #different cuts can give the same fragments
    #to use outlines to remove them
    outlines = set()

    if (mol == None):
        print('Mol is None')
        return None
    else:
        if design_task == "linker":
	        frags = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        elif design_task == "elaboration":
	        frags = rdMMPA.FragmentMol(mol, minCuts=1, maxCuts=1, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        else:
            print("Invalid choice for design_task. Must be 'linker' or 'elaboration'.")
        for core, chains in frags:
            if design_task == "linker":
                output = '%s,%s,%s,%s' % (smi, cid, core, chains)
            elif design_task == "elaboration":
                output = '%s,%s,%s' % (smi, cid, chains)
            if (not (output in outlines)):
                outlines.add(output)
        if not outlines:
            # for molecules with no cuts, output the parent molecule itself
            outlines.add('%s,%s,,' % (smi,cid))

    return outlines



def load_pp_file(file_path):
    node_type = []
    #node_size = []
    node_pos = []  # [(x,y,z)]

    lines = file_path.read_text().strip().split('\n')

    n_nodes = len(lines)

    assert n_nodes <= 7


    for line in lines:
        types, x, y, z = line.strip().split()
        
        tp = phar2idx.get(types, 0)

        node_type.append(tp)
        #node_size.append(size)
        node_pos.append(tuple(float(i) for i in (x, y, z)))

    node_type = np.array(node_type)
    #node_size = np.array(node_size)
    node_pos = np.array(node_pos)
    
    return node_type, node_pos


def load_ep_file(file_path):
   ## to be implemented
    return None



def prepare_pharma_data(sample_condition, n_nodes, bs):
    node_type, node_pos = load_phar_file(sample_condition)
    
    min_nodes = torch.min(n_nodes).item()
    
    assert min_nodes < len(node_type), "Error: Pharmacophore points are more than the number of atoms in the molecule"

        
    n = len(node_type)

    # Generate n random integers less than min_nodes
    random_numbers = np.random.randint(0, min_nodes, size=n)
    
    feat_array = np.zeros(min_nodes)
    coord_array = np.zeros((min_nodes, 3))
    mask_array = np.zeros(min_nodes)
    
    for i, idx in enumerate(random_numbers):
        feat_array[int(idx)] = node_type[i]
        coord_array[int(idx)] = node_pos[i]
        mask_array[int(idx)] = 1


    coord_array = torch.Tensor(coord_array).float()

    coord_array = coord_array - torch.mean(coord_array, dim=0, keepdim=True)
    pharma_feat = torch.Tensor(feat_array).long()
    mask_array = torch.tensor(mask_array).long()
    
    X = F.one_hot(pharma_feat, num_classes=len(FAMILY_MAPPING)+1).float()
    
    bs_coord_array = coord_array.repeat(bs, 1, 1)
    bs_X = X.repeat(bs, 1, 1)
    bs_mask_array = mask_array.repeat(bs, 1)
    
    print(bs_coord_array.shape, bs_X.shape, bs_mask_array.shape)
    
    return bs_coord_array, bs_X, bs_mask_array
    