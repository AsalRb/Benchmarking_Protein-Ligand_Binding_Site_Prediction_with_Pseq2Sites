# scPDB preprocessing

import os, warnings, subprocess, shutil
import numpy as np
import pandas as pd
from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBParser as _PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit import Chem
from biopandas.mol2 import PandasMol2

# ---------------------------
# 0) Paths
# ---------------------------
scpdb_path = r"C:\Users\Asal\Desktop\Pseq2Sites\Project\Datasets\Training\Before_preprocessing\scPDB_dataset\scPDB"
assert os.path.isdir(scpdb_path), f"Folder not found: {scpdb_path}"

after_root_base = r"C:\Users\Asal\Desktop\Pseq2Sites\Project\Datasets\Training\After_preprocessing"
after_root = os.path.join(after_root_base, "scPDB")
os.makedirs(after_root, exist_ok=True)

# Outputs
out_tsv_scid  = os.path.join(after_root, "scPDB_data.tsv")              # scPDB, Sequence, SMILES
out_tsv_pdbid = os.path.join(after_root, "scPDB_data_normalized.tsv")   # PDB, Sequence, SMILES

# Temp & cleaned
tmp_smiles_dir = os.path.join(after_root, "tmp_smiles")
cleaned_root   = os.path.join(after_root, "cleaned_PDBs_scPDB")
os.makedirs(tmp_smiles_dir, exist_ok=True)
os.makedirs(cleaned_root, exist_ok=True)

# Lists
list_all   = os.path.join(after_root, "scPDB_list_all.txt")
list_lenle = os.path.join(after_root, "scPDB_list_len_le_1500.txt")
list_final = os.path.join(after_root, "scPDB_list_final.txt")

print("scpdb_path:", scpdb_path, flush=True)
print("after_root:", after_root, flush=True)

# ---------------------------
# 1) Discover complexes
# ---------------------------
required = ("protein.mol2", "site.mol2", "ligand.mol2")
complex_list = sorted([
    d for d in os.listdir(scpdb_path)
    if os.path.isdir(os.path.join(scpdb_path, d))
       and all(os.path.isfile(os.path.join(scpdb_path, d, fn)) for fn in required)
])
print(f"[info] complexes found: {len(complex_list)}", flush=True)
print("[info] example:", complex_list[:10], flush=True)

with open(list_all, "w", encoding="utf-8") as f:
    f.write("\n".join(complex_list))
print(f"[done] wrote: {list_all}", flush=True)

# ---------------------------
# 2) protein.mol2 -> protein.pdb and remove HETATM
# ---------------------------
def mol2_to_pdb_if_needed(sid: str):
    mol2 = os.path.join(scpdb_path, sid, "protein.mol2")
    pdb  = os.path.join(scpdb_path, sid, "protein.pdb")
    if not os.path.isfile(pdb):
        if shutil.which("obabel") is None:
            raise RuntimeError("OpenBabel (obabel) not found on PATH")
        cmd = f'obabel -imol2 "{mol2}" -opdb -O "{pdb}"'
        subprocess.run(cmd, shell=True, check=True)

class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

warnings.simplefilter("ignore", PDBConstructionWarning)
parser = PDBParser(QUIET=True)

def clean_protein_write(sid: str):
    try:
        mol2_to_pdb_if_needed(sid)
        src_pdb = os.path.join(scpdb_path, sid, "protein.pdb")
        out_dir = os.path.join(cleaned_root, sid)
        os.makedirs(out_dir, exist_ok=True)
        dst_pdb = os.path.join(out_dir, f"{sid}_remove_HEATM_protein.pdb")

        structure = parser.get_structure(sid, src_pdb)
        io = PDBIO(); io.set_structure(structure)
        io.save(dst_pdb, NonHetSelect())
        return True
    except Exception as e:
        print(f"[clean err] {sid}: {e}", flush=True)
        return False

print(">> Step 2: convert + HETATM removal ...", flush=True)
ok = sum(clean_protein_write(sid) for sid in complex_list)
print(f"[done] Step 2 OK: {ok} / {len(complex_list)}", flush=True)

# Quick checks
# which cleaned PDBs are missing?
failed = [
    sid for sid in complex_list
    if not os.path.isfile(os.path.join(cleaned_root, sid, f"{sid}_remove_HEATM_protein.pdb"))
]
print("Failed count:", len(failed))
print("Failed IDs:", failed[:10])  # should show the single culprit

# Retry just those
for sid in failed:
    print("Retrying:", sid)
    _ = clean_protein_write(sid)

# Use this list from now on (Step 3+)
complex_list_ok = [
    sid for sid in complex_list
    if os.path.isfile(os.path.join(cleaned_root, sid, f"{sid}_remove_HEATM_protein.pdb"))
]
print("Proceeding with:", len(complex_list_ok), "complexes")

# ---------------------------
# 3) Parse sequences & binding indices
# ---------------------------
AA = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
      "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
      "THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O"}

pdb_parser = PDBParser(QUIET=True)

def _key(coord, ndigits=3):
    x, y, z = map(float, coord)
    return (round(x, ndigits), round(y, ndigits), round(z, ndigits))

def get_info_sc(sid: str):
    try:
        prot_file = os.path.join(cleaned_root, sid, f"{sid}_remove_HEATM_protein.pdb")
        site_file = os.path.join(scpdb_path, sid, "site.mol2")
        if not (os.path.isfile(prot_file) and os.path.isfile(site_file)):
            raise FileNotFoundError("missing cleaned protein or site.mol2")

        structure = pdb_parser.get_structure(sid, prot_file)
        chain_names, seqs, seq_lens = [], [], []
        coord_to_res, reindex = {}, 0

        model = structure[0]
        for chain_name in list(model.child_dict.keys()):
            chain = model[chain_name]
            seq = []
            for residue in chain.get_residues():
                if residue.resname in AA:
                    seq.append(AA[residue.resname])
                    for atom in residue:
                        coord_to_res[_key(atom.get_coord())] = reindex
                    reindex += 1
            if seq:
                chain_names.append(chain_name)
                seqs.append("".join(seq))
                seq_lens.append(len(seq))

        if not seqs:
            raise ValueError("no standard residues found in protein")

        pm2 = PandasMol2().read_mol2(site_file)
        sdf = pm2.df.copy()
        sdf = sdf.drop_duplicates("subst_name")
        sdf = sdf.loc[sdf["subst_name"].map(lambda a: isinstance(a, str) and a[:3] in AA)]

        if {"x","y","z"}.issubset(sdf.columns):
            coords = sdf[["x","y","z"]].values
        else:
            coords = sdf.iloc[:, [2,3,4]].values  # fallback for older biopandas

        binding_index = []
        for (x, y, z) in coords:
            k = _key((x, y, z))
            if k in coord_to_res:
                binding_index.append(coord_to_res[k])
        binding_index = sorted(set(binding_index))

        chain_str = ",".join(chain_names)
        seq_str   = ",".join(seqs)
        total_len = int(np.sum(np.array(seq_lens)))
        lens_str  = ",".join(map(str, seq_lens))
        bs_str    = ",".join(map(str, binding_index))
        return chain_str, seq_str, total_len, lens_str, bs_str
    except Exception as e:
        print(f"[get_info err] {sid}: {e}", flush=True)
        return None

print(">> Step 3: parsing sequences & binding sites ...", flush=True)
records = [get_info_sc(sid) for sid in complex_list_ok]
print("[done] Step 3", flush=True)

data_df = pd.DataFrame({"scPDB": complex_list_ok})
data_df["PDB"]               = data_df["scPDB"].map(lambda s: s[:4])
data_df["Chain"]             = [r[0] if r else None for r in records]
data_df["Sequence"]          = [r[1] if r else None for r in records]
data_df["Total_seq_lengths"] = [r[2] if r else None for r in records]
data_df["Chain_seq_lengths"] = [r[3] if r else None for r in records]
data_df["BS"]                = [r[4] if r else None for r in records]

data_df = data_df.loc[data_df["Sequence"].notna()].reset_index(drop=True)
data_df = data_df.loc[data_df["Chain"] != " "].reset_index(drop=True)
print(f"[info] after seq extraction: {len(data_df)} rows", flush=True)

# Quick checks
# Nothing else dropped?
print("Dropped since Step 2:",
      len(set(complex_list_ok) - set(data_df["scPDB"])))

# No empty sequences / zero lengths?
print("Empty sequences:", (data_df["Sequence"].str.len()==0).sum())
print("Zero total lengths:",
      (pd.to_numeric(data_df["Total_seq_lengths"], errors="coerce")<=0).sum())

# Chain lengths sum to total?
def _sum_lens(s):
    try: return sum(int(x) for x in str(s).split(",") if x)
    except: return None
mismatch = [i for i,(tot,lens) in enumerate(
    zip(data_df["Total_seq_lengths"], data_df["Chain_seq_lengths"])
) if _sum_lens(lens) != int(tot)]
print("Chain-lens ≠ total:", len(mismatch))

# Binding-site indices in range?
bad_bs = []
for i,(tot,bs) in enumerate(zip(data_df["Total_seq_lengths"], data_df["BS"])):
    if isinstance(bs,str) and bs:
        idxs = [int(x) for x in bs.split(",")]
        if any(j<0 or j>=int(tot) for j in idxs):
            bad_bs.append(i)
print("Out-of-range BS rows:", len(bad_bs))

# ---------------------------
# 4) Filter sequences > 1500 aa
# ---------------------------
lengths = pd.to_numeric(data_df["Total_seq_lengths"], errors="coerce").fillna(0).astype(int).values
data_df = data_df[lengths <= 1500].reset_index(drop=True)
print(f"[info] after length<=1500: {len(data_df)} rows", flush=True)

with open(list_lenle, "w", encoding="utf-8") as f:
    f.write("\n".join(data_df["scPDB"].tolist()))
print(f"[done] wrote: {list_lenle}", flush=True)

# Quick checks
# How many got removed?
print("Removed by length filter:", len(set(complex_list_ok)) - len(set(data_df["scPDB"])))

# Confirm the max is indeed ≤1500
print("Max length now:", int(pd.to_numeric(data_df["Total_seq_lengths"]).max()))

# ---------------------------
# 5) Ligand SMILES + filter ≤160
# ---------------------------
def read_lines(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return [i.strip() for i in f.readlines()]

def convert_smiles_by_sid(sid: str):
    mol2 = os.path.join(scpdb_path, sid, "ligand.mol2")
    out_file = os.path.join(tmp_smiles_dir, f"{sid}.smi")
    try:
        if shutil.which("obabel") is None:
            raise RuntimeError("OpenBabel (obabel) not found on PATH")
        cmd = f'obabel -imol2 "{mol2}" -osmi -xC | obabel -ismi -osmi -xk -O "{out_file}"'
        subprocess.run(cmd, shell=True, check=True)

        smiles = read_lines(out_file)[0].split("\t")[0].strip()
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None
        return MolToSmiles(mol, isomericSmiles=False, kekuleSmiles=True)
    except Exception as e:
        print(f"[SMILES err] {sid}: {e}", flush=True)
        return None

print(">> Step 5: ligand.mol2 -> SMILES ...", flush=True)
data_df["SMILES"] = [convert_smiles_by_sid(sid) for sid in data_df["scPDB"]]
data_df = data_df.loc[data_df["SMILES"].notna()].reset_index(drop=True)
keep = [len(s) <= 160 for s in data_df["SMILES"].values]
data_df = data_df.loc[keep].reset_index(drop=True)
print(f"[info] after SMILES<=160: {len(data_df)} rows", flush=True)

# Quick checks
# How many did we lose in Step 5 vs after Step 4?
print("Dropped in SMILES step:", 17103 - len(data_df))

# Any empty or None SMILES left? (should be 0)
print("Empty SMILES rows:", (data_df["SMILES"].str.len()==0).sum())

# RDKit sanity on a random sample
from rdkit import Chem
sample_bad = sum(Chem.MolFromSmiles(s) is None for s in data_df["SMILES"].sample(min(200, len(data_df)), random_state=0))
print("RDKit failures in sample:", sample_bad)

# .smi files present for a few examples (make sure tmp_smiles_dir still exists)
import os
subset = data_df["scPDB"].head(10).tolist()
print({sid: os.path.isfile(os.path.join(tmp_smiles_dir, f"{sid}.smi")) for sid in subset})

# ---------------------------
# 6) Save & cleanup
# ---------------------------
data_df.loc[:, ["scPDB", "Sequence", "SMILES"]].to_csv(out_tsv_scid, sep="\t", index=False)
print(f"[done] saved: {out_tsv_scid}", flush=True)

data_df.loc[:, ["PDB", "Sequence", "SMILES"]].to_csv(out_tsv_pdbid, sep="\t", index=False)
print(f"[done] saved: {out_tsv_pdbid}", flush=True)

with open(list_final, "w", encoding="utf-8") as f:
    f.write("\n".join(data_df["scPDB"].tolist()))
print(f"[done] saved final list: {list_final}", flush=True)

try:
    shutil.rmtree(tmp_smiles_dir)
    print(f"[cleanup] removed temporary folder: {tmp_smiles_dir}", flush=True)
except Exception as e:
    print(f"[cleanup] failed to remove tmp folder: {e}", flush=True)

print(data_df.head(5).to_string(index=False), flush=True)

# Quick checks
print("TSV (scPDB) exists:", os.path.isfile(out_tsv_scid))
print("TSV (PDB)   exists:", os.path.isfile(out_tsv_pdbid))
print("Rows in TSV should equal:", len(data_df))

