import sys
import random
import numpy as np

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcTPSA

# from libs.sas_scorer import *

atom_vocab = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
              'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
              'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
              'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']


def atom_feature(atom):
    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))

    return np.asarray(
        one_of_k_encoding_unk(atom.GetSymbol(), atom_vocab) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        [atom.GetIsAromatic()]
    )


def convert_smiles_to_graph(smi):
    mol = Chem.MolFromSmiles(smi.numpy())
    if mol is not None:
        feature = np.asarray([atom_feature(atom) for atom in mol.GetAtoms()])
        adj = np.asarray(Chem.rdmolops.GetAdjacencyMatrix(mol))
        graph = (feature, adj)
        return graph


def calc_properties(smi):
    # returns logP, TPSA, MW, MR
    m = Chem.MolFromSmiles(smi.numpy())
    logP = MolLogP(m)
    tpsa = CalcTPSA(m)
    # sas = calculateScore(m)
    mw = ExactMolWt(m)
    mr = MolMR(m)
    return np.asarray(logP), np.asarray(tpsa), np.asarray(mw), np.asarray(mr)


if __name__ == '__main__':
    smi_file = sys.argv[1]
    save_path = sys.argv[2]
    smi_list = open(smi_file, 'r').readlines()
    preprocess_inputs(smi_list, save_path)
