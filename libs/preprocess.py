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


LOGP_MEAN, LOGP_STD = 2.5451961055624057, 2.3098259954263725
MR_MEAN, MR_STD = 1.8846996475240654, 0.214787515251564
TPSA_MEAN, TPSA_STD = 1.7127172870304095, 0.4237963371365658
MW_MEAN, MW_STD = 2.4614866916977736, 0.20207222102872363


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
    # normalize quantities
    m = Chem.MolFromSmiles(smi.numpy())
    logP = np.asarray(MolLogP(m))
    logP = (logP - LOGP_MEAN) / LOGP_STD

    tpsa = np.asarray(CalcTPSA(m))
    tpsa = np.log10(tpsa + 1)
    tpsa = (tpsa - TPSA_MEAN) / TPSA_STD

    # sas = calculateScore(m)

    mw = np.asarray(ExactMolWt(m))
    mw = np.log10(mw + 1)
    mw = (mw - MW_MEAN) / MW_STD

    mr = np.asarray(MolMR(m))
    mr = np.log10(mr + 1)
    mr = (mr - MR_MEAN) / MR_STD
    return logP, tpsa, mw, mr


def logP_benchmark(smi):
    m = Chem.MolFromSmiles(smi.numpy())
    logP = MolLogP(m)
    return np.asarray(logP)


if __name__ == '__main__':
    smi_file = sys.argv[1]
    save_path = sys.argv[2]
    smi_list = open(smi_file, 'r').readlines()
    preprocess_inputs(smi_list, save_path)
