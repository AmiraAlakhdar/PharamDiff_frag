from __future__ import print_function
from rdkit import RDConfig, Chem, Geometry, DistanceGeometry
from rdkit.Chem import ChemicalFeatures, rdDistGeom, Draw, rdMolTransforms
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Chem.Draw import IPythonConsole, DrawingOptions
from rdkit.Numerics import rdAlignment
from rdkit import RDLogger
import os
import torch
from midi.datasets.pharmacophore_utils import get_features_factory
from collections import Counter


RDLogger.DisableLog('rdApp.*')

__FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))


PHARMACOPHORE_FAMILES_TO_KEEP = ('Aromatic', 'Hydrophobe', 'PosIonizable', 'Acceptor', 'Donor', 'LumpedHydrophobe')
FAMILY_MAPPING = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}
_FEATURES_FACTORY = []


def applyRadiiToBounds(radii,pcophore):
  for i in range(len(radii)):
    for j in range(i+1,len(radii)):
      sumRadii = radii[i]+radii[j]
      pcophore.setLowerBound(i,j,max(pcophore.getLowerBound(i,j)-sumRadii,0))
      pcophore.setUpperBound(i,j,pcophore.getUpperBound(i,j)+sumRadii)

def match_mol(mol, pharma_feat, pharma_coord, tolerance=1.21):
    
    unique_coords, unique_indices = torch.unique(pharma_coord, dim=0, return_inverse=True)
    pharma_coord = pharma_coord[unique_indices]
    pharma_feat = pharma_feat[unique_indices]
    
    
    Ph4Feats = []
    radii = []
    for i in range(len(pharma_feat)):
        feat = PHARMACOPHORE_FAMILES_TO_KEEP[int(pharma_feat[i])]
        g = Geometry.Point3D(pharma_coord[i, 0].item(), pharma_coord[i, 1].item(), pharma_coord[i, 2].item())
        Ph4Feats.append(ChemicalFeatures.FreeChemicalFeature(feat, g))
        radii.append(tolerance)
        
    pcophore = Pharmacophore.Pharmacophore(Ph4Feats)
    applyRadiiToBounds(radii,pcophore)
    
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        canMatch,allMatches = EmbedLib.MatchPharmacophoreToMol(mol,__FACTORY,pcophore)
        boundsMat = rdDistGeom.GetMoleculeBoundsMatrix(mol)
        
        if canMatch:
            failed,boundsMatMatched,matched,matchDetails = EmbedLib.MatchPharmacophore(allMatches,
                                                                                       boundsMat,
                                                                                       pcophore,
                                                                                       useDownsampling=False)
            if failed == 0:
                return 1
            else:
                return 0
        else:
            return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


def match_mol_2d(mol, pharma_feat, pharma_coord, tolerance=1.21):
    
    unique_coords, unique_indices = torch.unique(pharma_coord, dim=0, return_inverse=True)
    pharma_coord = pharma_coord[unique_indices]
    pharma_feat = pharma_feat[unique_indices]
    
    
    Ph4Feats = []
    radii = []
    for i in range(len(pharma_feat)):
        feat = PHARMACOPHORE_FAMILES_TO_KEEP[int(pharma_feat[i])]
        Ph4Feats.append(feat)
        
        
    feature_factory, keep_featnames = get_features_factory(PHARMACOPHORE_FAMILES_TO_KEEP)
    
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        feats = feature_factory.GetFeaturesForMol(mol, confId=-1)
    except:
        return -1

    features_in_mol = [feat.GetFamily() for feat in feats]
    
    
    feature_counts = Counter(features_in_mol)

    # Track matches
    matches = 0

    # For each target feature, check if it is present in the molecule counts
    for feature in Ph4Feats:
        if feature_counts[feature] > 0:
            matches += 1  # Count this feature as matched
            feature_counts[feature] -= 1  # Decrement the count in the molecule
    
    
    return matches / len(Ph4Feats)
