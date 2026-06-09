import pandas as pd

print("=== FINAL VERIFICATION OF N=5,739 ===")
final = pd.read_csv("submission_GROWN.csv", dtype=str)
print(f"Rows: {len(final):,}")
print(f"No duplicate BANC IDs: {final['BANC'].nunique() == len(final)}")
print(f"No duplicate FAFB IDs: {final['FAFB'].nunique() == len(final)}")
print(f"No duplicate MCNS IDs: {final['MCNS'].nunique() == len(final)}")

sample = final.sample(300, random_state=42)

def load_str(f):
    df = pd.read_csv(f, dtype=str, header=None)
    df.columns = ["src","tgt"]
    df["src"] = df["src"].str.strip()
    df["tgt"] = df["tgt"].str.strip()
    return df

fafb_df = load_str("fafb_783_edge_list.csv")
banc_df = load_str("banc_626_edge_list.csv")
mcns_df = load_str("mcns_0.9_edge_list.csv")

b2i = {b:i for i,b in enumerate(sample["BANC"])}
f2i = {f:i for i,f in enumerate(sample["FAFB"])}
m2i = {m:i for i,m in enumerate(sample["MCNS"])}

bi = banc_df[banc_df["src"].isin(b2i) & banc_df["tgt"].isin(b2i)]
fi = fafb_df[fafb_df["src"].isin(f2i) & fafb_df["tgt"].isin(f2i)]
mi = mcns_df[mcns_df["src"].isin(m2i) & mcns_df["tgt"].isin(m2i)]

bes = set(zip(bi["src"],bi["tgt"]))
fes = set(zip(fi["src"],fi["tgt"]))
mes = set(zip(mi["src"],mi["tgt"]))
all_e = bes | fes | mes

violations = sum(1 for e in all_e if not ((e in bes)==(e in fes)==(e in mes)))

print(f"\n300-neuron spot check:")
print(f"  Internal edge positions: {len(all_e):,}")
print(f"  Violations: {violations}")
if violations == 0:
    print("  RESULT: VALID - 0 violations confirmed!")
else:
    print(f"  RESULT: {violations} violations found!")

print("\n=== FINAL SUBMISSION FILE ===")
print("File: submission_GROWN.csv")
print(f"N = {len(final):,} neurons")
print("Columns: BANC, FAFB, MCNS")
print(final.head(5).to_string())
