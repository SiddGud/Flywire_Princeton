import pandas as pd
import time

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("CLAIM 1 — Bijection (one-to-one mapping)")
try:
    sub = pd.read_csv("submission_DEFINITIVE.csv", dtype=str)
    print(f"File loaded. Rows: {len(sub):,}")
    b_uniq = sub['BANC'].nunique() == len(sub)
    f_uniq = sub['FAFB'].nunique() == len(sub)
    m_uniq = sub['MCNS'].nunique() == len(sub)
    print(f"  a) All {len(sub):,} BANC values unique? {'YES' if b_uniq else 'NO'}")
    print(f"  b) All {len(sub):,} FAFB values unique? {'YES' if f_uniq else 'NO'}")
    print(f"  c) All {len(sub):,} MCNS values unique? {'YES' if m_uniq else 'NO'}")
except Exception as e:
    print(f"Error: {e}")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("CLAIM 2 — Isomorphism (most important)")
try:
    def load_str(f):
        df = pd.read_csv(f, dtype=str, header=None)
        df.columns = ["src","tgt"]
        df["src"]=df["src"].str.strip(); df["tgt"]=df["tgt"].str.strip()
        return df

    print("Loading edge lists...")
    fafb_df = load_str("fafb_783_edge_list.csv")
    banc_df = load_str("banc_626_edge_list.csv")
    mcns_df = load_str("mcns_0.9_edge_list.csv")

    samp = sub.sample(min(300, len(sub)), random_state=42)
    b_ids = samp["BANC"].tolist()
    f_ids = samp["FAFB"].tolist()
    m_ids = samp["MCNS"].tolist()
    
    banc_es = set(zip(banc_df["src"], banc_df["tgt"]))
    fafb_es = set(zip(fafb_df["src"], fafb_df["tgt"]))
    mcns_es = set(zip(mcns_df["src"], mcns_df["tgt"]))

    violations = 0
    edges_checked = 0
    for i in range(len(b_ids)):
        for j in range(len(b_ids)):
            if i == j: continue
            
            # Step 1, 2, 3: Check edge in EACH dataset's OWN ID space
            b_edge = (b_ids[i], b_ids[j]) in banc_es
            f_edge = (f_ids[i], f_ids[j]) in fafb_es
            m_edge = (m_ids[i], m_ids[j]) in mcns_es
            
            # Step 4: All three must agree
            if not (b_edge == f_edge == m_edge):
                violations += 1
            if b_edge or f_edge or m_edge:
                edges_checked += 1
                
    print(f"Checked 300 neurons.")
    print(f"Found {edges_checked} internal edge positions.")
    print(f"Exact violations count: {violations}")
except Exception as e:
    print(f"Error: {e}")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("CLAIM 3 — Dataset statistics")
try:
    fafb_uniq = set(fafb_df["src"]) | set(fafb_df["tgt"])
    banc_uniq = set(banc_df["src"]) | set(banc_df["tgt"])
    mcns_uniq = set(mcns_df["src"]) | set(mcns_df["tgt"])
    
    print(f"  a) Unique neurons in FAFB: {len(fafb_uniq):,}")
    print(f"  b) Unique neurons in BANC: {len(banc_uniq):,}")
    print(f"  c) Unique neurons in MCNS: {len(mcns_uniq):,}")
    
    print(f"  d) FAFB and BANC share IDs? {'YES' if len(fafb_uniq & banc_uniq) > 0 else 'NO'}")
    print(f"  e) FAFB and MCNS share IDs? {'YES' if len(fafb_uniq & mcns_uniq) > 0 else 'NO'}")
    print(f"  f) BANC and MCNS share IDs? {'YES' if len(banc_uniq & mcns_uniq) > 0 else 'NO'}")
except Exception as e:
    print(f"Error: {e}")

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("CLAIM 4 — IDs present in challenge files")
try:
    b_in = sum(sub["BANC"].isin(banc_uniq)) / len(sub) * 100
    f_in = sum(sub["FAFB"].isin(fafb_uniq)) / len(sub) * 100
    m_in = sum(sub["MCNS"].isin(mcns_uniq)) / len(sub) * 100
    print(f"  a) Fraction of BANC IDs in edge list: {b_in:.1f}%")
    print(f"  b) Fraction of FAFB IDs in edge list: {f_in:.1f}%")
    print(f"  c) Fraction of MCNS IDs in edge list: {m_in:.1f}%")
except Exception as e:
    print(f"Error: {e}")
