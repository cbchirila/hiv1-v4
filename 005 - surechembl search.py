from pathlib import Path
import re
import pandas as pd
import duckdb
from FPSim2 import FPSim2Engine

# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path("e:/x-hiv1-v4-data/")
SMI_FILE = BASE_DIR / "input - compounds/123cpds+32appHIV.smi"

SURECHEMBL_DIR = BASE_DIR / "input - surechembl"
COMPOUNDS_FILE = SURECHEMBL_DIR / "compounds.parquet"
FPS_FILE = SURECHEMBL_DIR / "fpsim2_fingerprints.h5"
PATENT_COMPOUND_FILE = SURECHEMBL_DIR / "patent_compound_map.parquet"
PATENTS_FILE = SURECHEMBL_DIR / "patents.parquet"

TANIMOTO_THRESHOLD = 0.50   # threshold sent to FPSim2 (>=)
TC_MIN_EXCLUSIV = 0.50      # we keep only TC > 0.50
TC_MAX_EXCLUSIV = 1.0       # we keep only TC < 1 (eliminates the identical compounds)

TITLE_KEYWORDS_GENERAL= [
    "viral", "virus", "virion", "virucid", "viremia", "viraemia", "virostatic",
    "oncolytic", "bacteriophage", "phage", "capsid", "nucleocapsid", "envelope", 
    "glycoprotein", "spike protein", "reverse transcriptase", "polymerase",
    "attenuated", "inactivated", "recombinant", "pseudotyped", "vector"
]

TITLE_KEYWORDS_DISEASES_FAMILIES = [
    "hepatitis", "immunodeficiency", "influenza", "herpes", "zoster", "varicella",
    "coronavirus", "covid", "sars", "dengue", "zika", "ebola", "chikungunya",
    "yellow fever", "measles", "rabies", "polio", "smallpox", "monkeypox", "mpox",
    "papilloma", "cytomegalo", "rhino", "enterovir", "picornavir", "filovir",
    "arenavir", "bunyavir", "paramyxo", "orthomyxo", "poxvir", "adenovir", "adenovirus", "norovir", "norovirus", "rotavir", "rotavirus", 
    "west nile", "marburg", "lassa", "hantavir", "polyomavir", "parvovir", 
    "retrovir", "lentivir", "flavivir", "togavir", "calicivir", "astrovir",
    "hepevir", "hepadnavir", "poxviridae", "retroviridae", "coronaviridae"
]

TITLE_KEYWORDS = TITLE_KEYWORDS_GENERAL + TITLE_KEYWORDS_DISEASES_FAMILIES

# Short acronyms: searched as whole words, not to match "archive" for HIV, not case sensitive.
TITLE_WORDS = [
    "HIV", "HBV", "HCV", "HDV", "HEV", "HAV", "HSV", "CMV", "RSV", "EBV", "HPV",
    "VZV", "SARS", "CoV", "COVID", "MERS", "flu", "HTLV", "SIV", "WNV", "AAV"
]

MAX_RESULTS = None        # None = no limit

# None = all compounds from the file (123 + 32)
# 123  = only compounds with numeric ID <= 123
MAX_QUERY_ID = None

# how many compounds are to be taken from the begining of the .smi file
# 123 = just the first 123 lines; None = all lines
MAX_LINES = 123

OUTPUT_FILE = BASE_DIR / "output" / "025 - surechembl search.csv.gz"
OUTPUT_HITS = BASE_DIR / "output" / "025 - surechembl hits.csv.gz"  # intermediate file, no patents

def make_name(raw):
    # ID numerical -> Compound_N ; ID text -> kept as such
    return f"Compound_{raw}" if re.fullmatch(r"\d+", raw) else raw

def get_cols(con,path):
    return list(con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')"
    ).fetchdf()["column_name"])

def pick(columns, candidates, contains=None):
    low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    if contains:
        for c in columns:
            if any(k in c.lower() for k in contains):
                return c
    return None

def main():
    for f in [COMPOUNDS_FILE, FPS_FILE, PATENT_COMPOUND_FILE, PATENTS_FILE, SMI_FILE]:
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    # ============================================================
    # READ SMILES  ("SMILES<TAB>ID", "SMILES ID" or tabel Markdown)
    # ============================================================
    records = []
    with open(SMI_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or re.match(r"^\|?\s*-+\s*\|", line):
                continue
            if "|" in line:
                parts = [x.strip() for x in line.strip("|").split("|")]
            else:
                parts = line.split()
            if len(parts) < 2:
                continue
            smiles, cid = parts[0], parts[1]
            if smiles.lower() in ("smiles", "smile"):
                continue
            records.append({"Query_Raw_ID": cid, "SMILES": smiles})

    query_df = pd.DataFrame(records).drop_duplicates("Query_Raw_ID").reset_index(drop=True)
    if query_df.empty:
        raise ValueError("No compounds in the .smi file")

    if MAX_QUERY_ID is not None:
        keep = query_df["Query_Raw_ID"].apply(
            lambda x: x.isdigit() and int(x) <= MAX_QUERY_ID
        )
        query_df = query_df[keep].reset_index(drop=True)

    if MAX_LINES:
        query_df = query_df.head(MAX_LINES).reset_index(drop=True)

    query_df["ID_compound"] = query_df["Query_Raw_ID"].apply(make_name)
    query_df["Order"] = range(len(query_df))  # the order from the file, for the final sorting
    print(f"Compound query: {len(query_df)}")
    print(query_df.head())

    # ============================================================
    # FPSIM2 SEARCH
    # ============================================================
    print("\nLoading database FPSim2 (~1.3 GB)...")
    engine = FPSim2Engine(str(FPS_FILE))
    try:
        print("Fingerprint type in database:", engine.fp_type, engine.fp_params)
    except Exception:
        pass

    # SureCHEMBL compounds that we want to follow (only digits)
    # Printing if found by FPSim2 and the Tanimoto similarity. E.g.: [379403]
    DEBUG_MOL_IDS = [379403]

    all_hits = []
    for i, row in enumerate(query_df.itertuples(index=False), start=1):
        try:
            # FPSim2 receives SMILES directly and computes the fingerprint
            res = engine.similarity(row.SMILES, TANIMOTO_THRESHOLD, n_workers=4)
        except Exception as e:
            print(f"[{i}/{len(query_df)}] {row.ID_compound}: error ({e}) was skipped")
            continue

        raw = pd.DataFrame({
            "mol_id": res["mol_id"].astype("int64"),
            "coeff": res["coeff"].astype(float),
        })
        n_all = len(raw)
        n_ident = int((raw["coeff"] >= TC_MAX_EXCLUSIV).sum())
        for dbg in DEBUG_MOL_IDS:
            found = raw[raw["mol_id"] == dbg]
            if len(found):
                print(f"[DEBUG] SCHEMBL{dbg} found for {row.ID_compound}:"
                    f"TC = {float(found['coeff'].iloc[0]):.4f}")
        
        # filering raw value (before rounding): 0.50 < TC < 1
        raw = raw[(raw["coeff"] > TC_MIN_EXCLUSIV) & (raw["coeff"] < TC_MAX_EXCLUSIV)]
        n = len(raw)
        print(f"[{i}/{len(query_df)}] {row.ID_compound}: {n} hits")
        print(f"(brute FPSim2: {n_all}, identical TC=1: {n_ident}")
        print(f"eliminated in total: {n_all - n})")
        if n == 0:
            continue

        h = pd.DataFrame({"mol_id": raw["mol_id"],"Tanimoto": raw["coeff"].round(4)})
        h = h.sort_values("Tanimoto", ascending=False)
        if MAX_RESULTS:
            h = h.head(MAX_RESULTS)
        h["ID_compound"] = row.ID_compound
        h["SMILES"] = row.SMILES
        h["Order"] = row.Order
        all_hits.append(h)

    if not all_hits:
        raise SystemExit(f"No Tanimoto hit >= {TANIMOTO_THRESHOLD}")

    hits_df = pd.concat(all_hits, ignore_index=True)
    hits_df.drop(columns="Order").to_csv(OUTPUT_HITS, index=False, encoding="utf-8-sig",compression="gzip")
    print(f"\nTotal hits: {len(hits_df)}")

    # ============================================================
    # SCHEMA PARQUET (join in DuckDB, without loading all in RAM)
    # ============================================================
    con = duckdb.connect()

    comp_cols = get_cols(con,COMPOUNDS_FILE)
    map_cols = get_cols(con,PATENT_COMPOUND_FILE)
    pat_cols = get_cols(con,PATENTS_FILE)

    print("compounds.parquet:", comp_cols)
    print("patent_compound_map.parquet:", map_cols)
    print("patents.parquet:", pat_cols)

    map_comp = pick(map_cols, ["schembl_id", "surechembl_id", "compound_id", "id"])
    map_pat = pick(map_cols, ["patent_id", "document_id", "publication_id", "patent"])
    pat_id = pick(pat_cols, ["patent_id", "id", "document_id", "publication_id"])
    pat_no = pick(pat_cols, ["patent_number", "publication_number", "patent_no", "document_number"], contains=["number", "publication"])
    pat_title = pick(pat_cols, ["title"], contains=["title"])

    print(f"map compound col: {map_comp}")
    print(f"map patent col: {map_pat}")
    print(f"patents id col: {pat_id}")
    print(f"patents number col: {pat_no}")
    print(f"patents title col: {pat_title}")

    missing = [n for n, v in [("map_comp", map_comp), ("map_pat", map_pat), ("pat_id", pat_id)] if v is None]
    if missing:
        raise ValueError(f"Rows not found: {missing} - check the schema that was printed above")

    q_map_comp = f'"{map_comp}"'
    q_map_pat = f'"{map_pat}"'
    q_pat_id = f'"{pat_id}"'

    # normalizing: keeping just digits (it works both with 'SCHEMBL123' and with 123)
    cid_expr = (f"TRY_CAST(regexp_replace(CAST({q_map_comp} AS VARCHAR), '[^0-9]', '', 'g') AS BIGINT)")
    no_expr = f'p."{pat_no}"' if pat_no else "NULL"
    title_expr = f'p."{pat_title}"' if pat_title else "NULL"

    con.register("hits", hits_df)

    if TITLE_KEYWORDS or TITLE_WORDS:
        if pat_title is None:
            raise ValueError("There is no title column in patents.parquet - cannot filter by title.")
        conds = []
        for kw in TITLE_KEYWORDS:
            kw_safe = kw.replace("'", "''")
            conds.append(f"p.\"{pat_title}\" ILIKE '%{kw_safe}%'")
        if TITLE_WORDS:
            words = "|".join(re.escape(w) for w in TITLE_WORDS)
            pattern = (r"\b(" + words + r")\b").replace("'", "''")
            conds.append(f"regexp_matches(p.\"{pat_title}\", '{pattern}', 'i')")
        where_clause = "WHERE " + " OR ".join(conds)
        join_type = "JOIN"        # just the hits that have patent in title
    else:
        where_clause = ""
        join_type = "LEFT JOIN"

    sql = f"""
    WITH m AS (
        SELECT {cid_expr} AS cid,
            CAST({q_map_pat} AS VARCHAR) AS pid
        FROM read_parquet('{PATENT_COMPOUND_FILE.as_posix()}')
        WHERE {cid_expr} IN (SELECT mol_id FROM hits)
    )
    SELECT
        h.ID_compound,
        h.SMILES,
        'SCHEMBL' || CAST(h.mol_id AS VARCHAR) AS SCHEMBL_ID,
        h.Tanimoto,
        {no_expr}    AS Patent_No,
        {title_expr} AS Patent_Title,
        h."Order"    AS Ord
    FROM hits h
    {join_type} m ON m.cid = h.mol_id
    {join_type} read_parquet('{PATENTS_FILE.as_posix()}') p
        ON CAST(p.{q_pat_id} AS VARCHAR) = m.pid
    {where_clause}
    """

    print("\nJoin with patents (DuckDB)...")
    final_df = con.execute(sql).fetchdf().drop_duplicates()

    if final_df.empty:
        con.close()
        raise SystemExit(f"No patent with the words from TITLE_KEYWORDS {TITLE_KEYWORDS} in title for the found hits.")

    final_df = (final_df.sort_values(["Ord", "Tanimoto"], ascending=[True, False]).drop(columns="Ord"))
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Final rows: {len(final_df)}")
    print(f"File: {OUTPUT_FILE}")
    print(f"Similarity hits: {OUTPUT_HITS}")
    print()
    print(final_df.head(20).to_string(index=False))
    con.close()
    return

if __name__=="__main__":
    main()
