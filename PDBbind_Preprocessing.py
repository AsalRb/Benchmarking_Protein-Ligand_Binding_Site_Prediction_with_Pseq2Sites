# PDBbind preprocessing

import os, warnings, subprocess, shutil
import numpy as np
import pandas as pd
from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBParser as _PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles

# ---------------------------
# 0) Paths (edit if needed)
# ---------------------------
pdb_path = r"C:\Users\Asal\Desktop\Pseq2Sites\Project\Datasets\Training\Before_preprocessing\PDBbind_dataset\PDBbind"
assert os.path.isdir(pdb_path), f"Folder not found: {pdb_path}"

# Central output folder for everything after preprocessing
after_path = r"C:\Users\Asal\Desktop\Pseq2Sites\Project\Datasets\Training\After_preprocessing"
os.makedirs(after_path, exist_ok=True)

# final TSV and temp/cleaned folders under after_path
out_tsv = os.path.join(after_path, "PDBbind_data.tsv")
tmp_smiles_dir = os.path.join(after_path, "tmp_smiles")
os.makedirs(tmp_smiles_dir, exist_ok=True)
cleaned_root = os.path.join(after_path, "cleaned_PDBs")
os.makedirs(cleaned_root, exist_ok=True)

# keep a copy of the list next to outputs
list_file = os.path.join(after_path, "PDBbind_list.txt")

# info_path (original parent) still available if needed
info_path = os.path.dirname(pdb_path)

# quick prints
print("pdb_path:", pdb_path, flush=True)
print("after_path (outputs):", after_path, flush=True)
print("out_tsv:", out_tsv, flush=True)
print("tmp_smiles_dir:", tmp_smiles_dir, flush=True)
print("cleaned_root:", cleaned_root, flush=True)

# write test to after_path
try:
    testfile = os.path.join(after_path, "._write_test")
    with open(testfile, "w") as f:
        f.write("ok")
    os.remove(testfile)
    print("write test to after_path: OK", flush=True)
except Exception as e:
    print("write test FAILED:", e, flush=True)

# ---------------------------
# 1) Build filtered complex list
# ---------------------------
required = ("_protein.pdb", "_ligand.mol2", "_pocket.pdb")
complex_list = sorted([
    d for d in os.listdir(pdb_path)
    if len(d) == 4
    and os.path.isdir(os.path.join(pdb_path, d))
    and all(os.path.isfile(os.path.join(pdb_path, d, f"{d}{suf}")) for suf in required)
])

print(f"[info] Complexes found: {len(complex_list)}", flush=True)
print("[info] Example IDs:", complex_list[:10], flush=True)

# duplicate check & save list into after_path
dups_ok = len(complex_list) == len(set(complex_list))
print(f"[info] duplicates check OK: {dups_ok}", flush=True)

with open(list_file, "w", encoding="utf-8") as f:
    f.write("\n".join(complex_list))
print(f"[done] Complex list saved to: {list_file}", flush=True)

# ---------------------------
# 2) Remove HETATM -> write cleaned pdbs under cleaned_root
# ---------------------------
def remove_HEATM_PDBbind(input_list, pdb_path, cleaned_root):
    class NonHetSelect(Select):
        def accept_residue(self, residue):
            # keep only standard residues (id[0] == " ")
            return 1 if residue.id[0] == " " else 0

    warnings.simplefilter("ignore", PDBConstructionWarning)
    parser = PDBParser(QUIET=True)

    saved, skipped = 0, 0
    for pid in input_list:
        src_file = os.path.join(pdb_path, pid, f"{pid}_protein.pdb")
        out_dir_pid = os.path.join(cleaned_root, pid)
        os.makedirs(out_dir_pid, exist_ok=True)
        des_file = os.path.join(out_dir_pid, f"{pid}_remove_HEATM_protein.pdb")

        if not os.path.isfile(src_file):
            print(f"[skip] {pid}: missing _protein.pdb", flush=True)
            skipped += 1
            continue
        try:
            structure = parser.get_structure(pid, src_file)
            io = PDBIO(); io.set_structure(structure)
            io.save(des_file, NonHetSelect())
            saved += 1
        except Exception as e:
            print(f"[err] {pid}: {e}", flush=True)
    print(f"[done] HEATM removal: saved={saved}, skipped={skipped}", flush=True)

print(">> Step 2: removing HETATM and writing cleaned PDBs ...", flush=True)
remove_HEATM_PDBbind(complex_list, pdb_path, cleaned_root)
print(">> Step 2 finished", flush=True)

# Quick checks
import os
pids = complex_list[:5]
for pid in pids:
    path = os.path.join(cleaned_root, pid, f"{pid}_remove_HEATM_protein.pdb")
    print(pid, os.path.isfile(path))

# ---------------------------
# 3) Extract sequences and binding indices from cleaned PDBs and pocket files
# ---------------------------
AA = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G",
      "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
      "THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O"}

pdb_parser = _PDBParser(QUIET=True)

def _key(coord, ndigits=3):
    x, y, z = map(float, coord)
    return (round(x, ndigits), round(y, ndigits), round(z, ndigits))

def get_info(pid, cleaned_root, pdb_path):
    try:
        prot_file = os.path.join(cleaned_root, pid, f"{pid}_remove_HEATM_protein.pdb")
        pocket_file = os.path.join(pdb_path, pid, f"{pid}_pocket.pdb")

        if not (os.path.isfile(prot_file) and os.path.isfile(pocket_file)):
            raise FileNotFoundError("required protein/pocket file missing")

        structure = pdb_parser.get_structure(pid, prot_file)
        chain_names, seqs, seq_lens = [], [], []
        coord_to_res = {}
        reindex = 0

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
            raise ValueError("no standard residues found")

        pocket = pdb_parser.get_structure(pid, pocket_file)
        binding_index = []
        for chain_name in list(pocket[0].child_dict.keys()):
            chain = pocket[0][chain_name]
            for residue in chain.get_residues():
                if residue.resname in AA:
                    for atom in residue:
                        k = _key(atom.get_coord())
                        if k in coord_to_res:
                            binding_index.append(coord_to_res[k])

        binding_index = sorted(set(binding_index))
        chain_str = ",".join(chain_names)
        seq_str = ",".join(seqs)
        total_len = int(np.sum(np.array(seq_lens)))
        lens_str = ",".join(map(str, seq_lens))
        bs_str = ",".join(map(str, binding_index))
        return chain_str, seq_str, total_len, lens_str, bs_str
    except Exception as e:
        print(f"[get_info err] {pid}: {e}", flush=True)
        return None

print(">> Step 3: parsing cleaned PDBs and pockets ...", flush=True)
records = []
for pid in complex_list:
    rec = get_info(pid, cleaned_root, pdb_path)
    records.append(rec)
print(">> Step 3 finished", flush=True)

data_df = pd.DataFrame({"PDB": complex_list})
data_df["Chain"] = [r[0] if r else None for r in records]
data_df["Sequence"] = [r[1] if r else None for r in records]
data_df["Total_seq_lengths"] = [r[2] if r else None for r in records]
data_df["Chain_seq_lengths"] = [r[3] if r else None for r in records]
data_df["BS"] = [r[4] if r else None for r in records]

# drop entries with missing sequence
data_df = data_df.loc[data_df["Sequence"].notna()].reset_index(drop=True)
data_df = data_df.loc[data_df["Chain"] != " "].reset_index(drop=True)

print(f"[info] after seq extraction: {len(data_df)} complexes", flush=True)

# Quick checks
# 1) Which PDBs were dropped?
dropped = sorted(set(complex_list) - set(data_df["PDB"]))
print("Dropped:", len(dropped), dropped[:10])

# 2) Any empty sequences or zero lengths lingering? (should be 0)
print("Empty seq rows:", (data_df["Sequence"].str.len()==0).sum())
print("Zero total lengths:", (pd.to_numeric(data_df["Total_seq_lengths"])<=0).sum())

# 3) Do chain lengths sum to total?
def _sum_lens(s):
    try:
        return sum(int(x) for x in str(s).split(",") if x)
    except:
        return None
mismatch = []
for i, (tot, lens) in enumerate(zip(data_df["Total_seq_lengths"], data_df["Chain_seq_lengths"])):
    if _sum_lens(lens) != int(tot):
        mismatch.append(i)
print("Chain-lens≠total count:", len(mismatch))

# 4) Are binding-site indices within sequence length?
bad_bs = []
for i, (tot, bs) in enumerate(zip(data_df["Total_seq_lengths"], data_df["BS"])):
    if isinstance(bs, str) and bs:
        idxs = [int(x) for x in bs.split(",")]
        if any((j < 0 or j >= int(tot)) for j in idxs):
            bad_bs.append(i)
print("Out-of-range BS rows:", len(bad_bs))

# 5) Peek a couple rows
print(data_df.head(3).to_string(index=False))

# ---------------------------
# 4) Filter sequences longer than 1500 aa
# ---------------------------
lengths = pd.to_numeric(data_df["Total_seq_lengths"], errors="coerce").fillna(0).astype(int).values
data_df = data_df[lengths <= 1500].reset_index(drop=True)
print(f"[info] after length<=1500: {len(data_df)} complexes", flush=True)

# Quick checks
# How many were removed and which?
removed_len = sorted(set(data_df["PDB"]).symmetric_difference(set(complex_list)))
print("Removed by length filter:", len(removed_len))

# Confirm no sequences >1500 remain
print("Max length now:", int(pd.to_numeric(data_df["Total_seq_lengths"]).max()))

# Peek the 5 longest still kept (should all be <=1500)
print(
    data_df.sort_values("Total_seq_lengths", ascending=False)
           .head(5)[["PDB","Total_seq_lengths","Chain_seq_lengths"]]
           .to_string(index=False)
)

# Saving the filtered PDB list for reproducibility
from pathlib import Path

filtered_list_file = Path(after_path) / "PDBbind_list_len_le_1500.txt"
with filtered_list_file.open("w", encoding="utf-8") as f:
    f.write("\n".join(data_df["PDB"].tolist()))
print("[done] saved:", filtered_list_file)

# ---------------------------
# 5) Ligand SMILES (OpenBabel -> RDKit) + filter SMILES length<=160
# ---------------------------
def read_lines(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return [i.strip() for i in f.readlines()]

def convert_smiles_by_pid(pid):
    mol2 = os.path.join(pdb_path, pid, f"{pid}_ligand.mol2")
    out_file = os.path.join(tmp_smiles_dir, f"{pid}.smi")
    try:
        if shutil.which("obabel") is None:
            raise RuntimeError("OpenBabel (obabel) not found on PATH")

        # Mol2 -> SMILES (strip coords) -> canonicalize skeletal
        cmd = f'obabel -imol2 "{mol2}" -osmi -xC | obabel -ismi -osmi -xk -O "{out_file}"'
        subprocess.run(cmd, shell=True, check=True)

        smiles = read_lines(out_file)[0].split("\t")[0].strip()
        smiles = MolToSmiles(MolFromSmiles(smiles), isomericSmiles=False, kekuleSmiles=True)
        return smiles
    except Exception as e:
        print(f"[SMILES err] {pid}: {e}", flush=True)
        return None

print(">> Step 5: converting ligands to SMILES (this can take a while) ...", flush=True)
smiles_list = []
for pid in data_df["PDB"].tolist():
    smi = convert_smiles_by_pid(pid)
    smiles_list.append(smi)
data_df["SMILES"] = smiles_list
data_df = data_df.loc[data_df["SMILES"].notna()].reset_index(drop=True)
print(f"[info] after SMILES parse: {len(data_df)} complexes", flush=True)

# SMILES length filter
keep = [len(s) <= 160 for s in data_df["SMILES"].values]
data_df = data_df.loc[keep].reset_index(drop=True)
print(f"[info] after SMILES<=160: {len(data_df)} complexes", flush=True)

# Quick checks
# 1) Basic tallies
print("Rows after SMILES parse & filter:", len(data_df))
print("Empty SMILES rows:", (data_df["SMILES"].str.len()==0).sum())
print("SMILES length (min/mean/max):",
      int(data_df["SMILES"].str.len().min()),
      round(data_df["SMILES"].str.len().mean(), 2),
      int(data_df["SMILES"].str.len().max()))

# 2) Inspect a few rows (spot check)
print(data_df.sample(5, random_state=0)[["PDB","SMILES"]].to_string(index=False))
print("\nTop 5 longest kept (should be ≤160):")
print(
    data_df.assign(L=data_df["SMILES"].str.len())
           .sort_values("L", ascending=False)
           .head(5)[["PDB","L","SMILES"]]
           .to_string(index=False)
)

# 3) Which PDBs failed SMILES parsing (dropped between Step 4 and Step 5)?
import os
step4_list_file = os.path.join(after_path, "PDBbind_list_len_le_1500.txt")
with open(step4_list_file, "r", encoding="utf-8") as f:
    step4_ids = [ln.strip() for ln in f if ln.strip()]

kept_ids = set(data_df["PDB"])
failed_ids = sorted(set(step4_ids) - kept_ids)
print("Failed SMILES count:", len(failed_ids))
print("First 20 failed:", failed_ids[:20])

# 4) Do we have .smi files for kept PDBs?
import os
subset = data_df["PDB"].head(20).tolist()
exists_map = {pid: os.path.isfile(os.path.join(tmp_smiles_dir, f"{pid}.smi")) for pid in subset}
print("tmp_smiles presence (first 20):", exists_map)

# Count how many kept PDBs have .smi on disk
have_smi = sum(os.path.isfile(os.path.join(tmp_smiles_dir, f"{pid}.smi")) for pid in data_df["PDB"])
print(f".smi files present for kept:", have_smi, "of", len(data_df))

# 5) RDKit sanity on a sample
from rdkit import Chem
sample = data_df.sample(min(200, len(data_df)), random_state=1)
bad = [(pid, smi) for pid, smi in zip(sample["PDB"], sample["SMILES"]) if Chem.MolFromSmiles(smi) is None]
print("RDKit failures in sample:", len(bad))
print("First few failures:", bad[:5])

# ---------------------------
# 6) Save TSV and cleanup
# ---------------------------
data_df[["PDB", "Sequence", "SMILES"]].to_csv(out_tsv, sep="\t", index=False)
print(f"[done] Saved: {out_tsv}", flush=True)

# remove temporary smiles folder
try:
    shutil.rmtree(tmp_smiles_dir)
    print(f"[cleanup] removed temporary folder: {tmp_smiles_dir}", flush=True)
except Exception as e:
    print(f"[cleanup] failed to remove tmp folder: {e}", flush=True)

# peek first rows
print(data_df.head(5).to_string(index=False), flush=True)

# Saving final PDB ID list file
final_list = os.path.join(after_path, "PDBbind_list_final.txt")
with open(final_list, "w", encoding="utf-8") as f:
    f.write("\n".join(data_df["PDB"].tolist()))
print("[done] Saved final list:", final_list)

# Quick checks
import os
print("TSV exists:", os.path.isfile(out_tsv))
print("Rows in TSV should equal:", len(data_df))

