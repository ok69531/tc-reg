import os
import pickle
import logging
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint, Descriptors

logger = logging.getLogger(__name__)


def smiles2fing(mol):
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=int)
    morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), dtype=int)
    rdkit = np.array(RDKFingerprint(mol), dtype=int)
    layered = np.array(AllChem.LayeredFingerprint(mol), dtype=int)
    pattern = np.array(AllChem.PatternFingerprint(mol), dtype=int)
    descriptor = Descriptors.CalcMolDescriptors(mol)
    
    fingerprints = {
        'maccs': maccs,
        'morgan': morgan,
        'rdkit': rdkit,
        'layered': layered,
        'pattern': pattern,
        'descriptor': descriptor
    }
    
    return fingerprints


def generate_features(root = '../dataset/raw'):
    file_name = 'tc_data.xlsx'
    df = pd.read_excel(os.path.join(root, file_name))
    with open(os.path.join(root, 'assays.txt'), 'r') as f:
        assays = f.read().split('\n')

    raw_y = df[['DTXSID', 'SMILES'] + assays]
    raw_smiles = df.SMILES
    raw_dtxsid = df.DTXSID
    raw_mols = [Chem.MolFromSmiles(x) for x in raw_smiles]
    
    mol_idx = [bool(x) for x in raw_mols]

    smiles = raw_smiles[mol_idx]
    dtxsids = raw_dtxsid[mol_idx]
    mols = list(filter(None, raw_mols))

    feat_dict = {x: None for x in dtxsids}
    for dtxsid, smi, mol in zip(dtxsids, smiles, mols):
        item = {
            'smiles': smi
        }
        item.update(smiles2fing(mol))
        feat_dict[dtxsid] = item
    
    processed_data = {
        'mols': mol_idx,
        'x': feat_dict,
        'df': raw_y,
    }
    
    save_path = f'../dataset/processed'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f'ml_tc.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)
    
    return processed_data


def load_dataset(*, root = '../dataset', assay_name = None, fp_type = 'maccs', log_transform = True):
    file_name = f'ml_tc.pkl'
    data_path = os.path.join(root, 'processed', file_name)
    
    if os.path.exists(data_path):
        logger.info('File Exsit')
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        logger.info('File not Exist')
        logger.info('Preprocessing data...')
        dataset = generate_features(root = os.path.join(root, 'raw'))
    
    cols = ['DTXSID', 'SMILES'] + [assay_name]
    df = dataset['df'][cols]
    
    mols = dataset['mols']
    df = df[mols]
    
    df = df.dropna().reset_index(drop = True)
    
    y = df[assay_name].to_numpy()
    if log_transform:
        y = np.log10(y)
    smiles = df.SMILES.tolist()
    
    fingerprints, descriptors = [], []
    for dtxsid in df.DTXSID:
        fingerprints.append(dataset['x'][dtxsid][fp_type])
        descriptors.append(dataset['x'][dtxsid]['descriptor'])
    
    fingerprints = np.stack(fingerprints)
    descriptors = pd.DataFrame(descriptors)
    valid_cols = descriptors.columns[~descriptors.isna().any(axis=0)]
    descriptors = descriptors[valid_cols]
    descriptors = descriptors.to_numpy()
    
    x = np.concatenate([fingerprints, descriptors], axis = 1)
    
    return x, y, smiles


if __name__ == '__main__':
    dataset = load_dataset(assay_name = 'ACEA_AR_agonist_80hr')
    