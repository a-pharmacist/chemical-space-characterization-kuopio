import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from rdkit.Chem.rdchem import Atom
from rdkit.Chem.MolStandardize import rdMolStandardize


def select_cols(file, col_id="Molecule ChEMBL ID", col_smiles="Smiles",
                col_mw="Molecular Weight", col_relation="Standard Relation",
                col_value="Standard Value", col_unit="Standard Units",
                threshold=10000):
                
    """Process data

    Args:
        file (str/pd.DataFrame): Path to file or pandas dataframe
        col_id (str): ID column name
        col_smiles (str): SMILES column name
        col_mw (str): Molecular weight column name
        col_relation (str): Relation column name
        col_value (str): Values column name
        col_unit (str): Unit column name
        threshold (int): Binary threshold in nM

    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Read file and prepare dataframe
    if isinstance(file, str):
        raw_df = pd.read_csv(file, quotechar='"', sep=";", low_memory=False)
        # raw_df = pd.read_csv(file, quotechar='"', sep=",", low_memory=False)
    else:
        raw_df = file.copy(deep=True)
    df = raw_df[[col_id, col_smiles, col_mw, col_relation,
                 col_value, col_unit]]

    df = df.dropna()

    # Get only = relation
    # df = df[df[col_relation] == "'='"]

    # Select only supported values format
    df = df[df[col_unit].isin(["ug.mL-1", "nM", "uM"])]
    weights = list(df[col_mw])
    values = list(df[col_value])
    units = list(df[col_unit])

    # Convert values
    converted_values = []
    for weight, value, unit in zip(weights, values, units):
        if unit == "ug.mL-1":
            val = (value / float(weight)) * 1000000
        elif unit == "uM":
            val = value * 1000
        elif unit == "nM":
            val = value

        converted_values.append(val)

    # Create activity lists
    df["Converted Value"] = converted_values
    df["Converted Units"] = ["nM"] * len(converted_values)

    conditions = [
        (df[col_relation] == "'='") & (df["Converted Value"] > threshold),
        (df[col_relation] == "'='") & (df["Converted Value"] < threshold),
        (df[col_relation] == "'<'") & (df["Converted Value"] > threshold),
        (df[col_relation] == "'<'") & (df["Converted Value"] < threshold),
        (df[col_relation] == "'<='") & (df["Converted Value"] > threshold),
        (df[col_relation] == "'<='") & (df["Converted Value"] < threshold),
        (df[col_relation] == "'>'") & (df["Converted Value"] > threshold),
        (df[col_relation] == "'>'") & (df["Converted Value"] < threshold),
        (df[col_relation] == "'>='") & (df["Converted Value"] > threshold),
        (df[col_relation] == "'>='") & (df["Converted Value"] < threshold)
    ]

    values = ["0", "1", "Invalid", "1", "Invalid", "1", "0", "Invalid", "0", "Invalid"]

    df["label"] = np.select(conditions, values)

    df = df[df["label"] != "Invalid"]

    df = df.rename(columns={"Converted Value": "raw activity"})

    output = df[[col_id, col_smiles, "raw activity", "label"]]
    output = output.rename(columns={col_id: "id", col_smiles: "smiles"})

    return output


def standardize_smiles(file, smiles_col="smiles", desc=""):
    """Standardize SMILES from file

    Args:
        file (str/pd.DataFrame): Path to file or pandas dataframe
        smiles_col (str): SMILES column name

    Returns:
        pd.DataFrame: Processed dataframe
    """
    text = f"SMILES standardization {desc}"

    # Read file and prepare dataframe
    if isinstance(file, str):
        raw_df = pd.read_csv(file)
        raw_df = raw_df.dropna(subset=[smiles_col])
    else:
        raw_df = file.dropna(subset=[smiles_col])
    df = raw_df.copy(deep=True)

    # Standardize smiles
    smiles = list(df[smiles_col])
    with BlockLogs():
        standardized = [_standardize(smi) for smi in tqdm(smiles, ncols=120,
                                                          desc=text)]
    clean_smiles = [x[0] for x in standardized]
    clean_inchi = [x[1] for x in standardized]

    # Preapare dataframe
    df["smiles_standardized"] = clean_smiles
    df["inchi"] = clean_inchi
    df = df.dropna(subset=["inchi"])

    # Get RDKit mol from standardized SMILES
    with BlockLogs():
        df_smiles = list(df["smiles_standardized"])
        mols = [Chem.MolFromSmiles(smi) for smi in df_smiles]
        df["mol"] = mols

    return df


def remove_duplicates(file, inchi_col="inchi", label_col="label",
                      exp_col="raw activity", desc=""):
    """Remove duplicated SMILES by inchi

    Args:
        file (str/pd.DataFrame): Path to file or pandas dataframe
        inchi_col (str): Inchi column name
        label_col (str): Label column name
        exp_col (str): Standardized dependent variable column name

    Returns:
        pd.DataFrame: Processed dataframe
    """
    text = f"Removing duplicates {desc}"

    # Read file and prepare dataframe
    if isinstance(file, str):
        raw_df = pd.read_csv(file)
    else:
        raw_df = file.copy(deep=True)
    df = raw_df.dropna(subset=[inchi_col])

    # Prepare
    unique_data = pd.DataFrame()
    task = len(df[label_col].unique())
    unique_hash = list(set(df[inchi_col]))

    # Check if experimental value column is present
    if exp_col in list(df.columns):
        target_col = exp_col
    else:
        target_col = label_col

    # Remove duplicates loop
    for hashs in tqdm(unique_hash, ncols=120, desc=text):
        selection = df[df[inchi_col] == hashs]
        n_labels = len(selection[label_col].unique())

        # Filter invalid mols
        with BlockLogs():
            smiles = selection.head(1)["smiles_standardized"].values[0]
            mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # Unique data
            if len(selection) == 1:
                unique_data = pd.concat([unique_data, selection], axis=0)

            # Duplicated data
            else:
                sel = selection.head(1).copy(deep=True)

                # Classification (same class)
                if (task == 2) and (n_labels == 1):
                    unique_data = pd.concat([unique_data, sel], axis=0)

                elif (task != 2):  # Regression
                    sel[target_col] = selection[target_col].mean()
                    unique_data = pd.concat([unique_data, sel], axis=0)

    # Export data
    output = unique_data.rename(columns={"smiles": "smiles_old",
                                         "smiles_standardized": "smiles"})
    output = output.drop(columns=["smiles_old", "inchi", exp_col],
                         errors="ignore")

    return output


def _standardize(smiles: str):
    """Standardize SMILES

    Source:
    https://github.com/PatWalters/practical_cheminformatics_tutorials/blob/main/misc/working_with_ChEMBL_drug_data.ipynb

    Args:
        smiles (str): Target SMILES

    Returns:
        tuple: Tuple with standardized SMILES and InChI Key
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:

        # flatten the molecule
        # smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        # mol = Chem.MolFromSmiles(smi)

        # removeHs, disconnect metal atoms, normalize and reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if  fragments, get the "parent" (the actual mol we are interested in)
        clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger()
        clean_mol = uncharger.uncharge(clean_mol)

        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        try:
            te = rdMolStandardize.TautomerEnumerator()  # idem
            clean_mol = te.Canonicalize(clean_mol)
        except RuntimeError:
            pass

        # There still may be duplicates that are the same structure
        # with different charge states. Generate an InChI for each
        # structure and remove the charge layer.
        inchi = Chem.MolToInchi(clean_mol)
        inchi = re.sub("/p\+[0-9]+", "", inchi)
        inchi_key = Chem.inchi.InchiToInchiKey(inchi)

        clean_smiles = Chem.MolToSmiles(clean_mol)

        # allow only molecules with C, N, O, F, P, S, Cl, Br, I
        atomic_no = [6, 7, 8, 9, 15, 16, 17, 35, 53]
        allowed_atoms = [Atom(i) for i in atomic_no]
        allow = rdMolStandardize.AllowedAtomsValidation(allowed_atoms)
        msg = allow.validate(clean_mol)

        # Remove molecules with no Carbon (C) atom
        if "c" not in clean_smiles.lower():
            msg.append("No carbon")

        if len(msg) != 0:
            clean_smiles = np.nan
            inchi_key = np.nan

    return (clean_smiles, inchi_key)
