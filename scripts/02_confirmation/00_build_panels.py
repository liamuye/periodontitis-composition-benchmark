#!/usr/bin/env python3
"""Step 1: build the frozen 15-panel confirmation table, pathway families and
the fixed 27-panel shared-axis reference, then record SHA-256.

No model is fitted here.  Outputs:
  01_panel_table.tsv
  01_panels_confirmation.json
  02_pathway_families.tsv
  03_shared_axis_reference.tsv
  provenance_inputs.json
  SHA256SUMS_premodel.txt
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_base

# ---------------------------------------------------------------- definitions
# gene order follows the confirmation-set source files; directions are the
# published directions explicitly stated in the PubMed abstract (0 = not
# stated, learned from the training-fold t sign).
PANELS = [
    dict(panel_id=101, pmid="39717623", year="2024", journal="Mediators Inflamm",
         pathway="Immune-related biomarkers", family="Immune and inflammatory",
         genes=["CD69", "CD93", "CXCL6"],
         directions=[0, 0, 0],
         source="survey_43_pool (row 1)",
         notes="Unsigned in abstract; oriented by training-fold Welch t sign."),
    dict(panel_id=102, pmid="41265166", year="2026", journal="Int Dent J",
         pathway="General transcriptomic diagnosis", family="Other",
         genes=["COL15A1", "CSF2RB", "CYP24A1", "MME", "NEFM"],
         directions=[0, 0, 0, 0, 0],
         source="survey_43_pool (row 2)",
         notes="5-gene classifier explicitly listed in abstract; unsigned."),
    dict(panel_id=103, pmid="42644235", year="2026", journal="Mediators Inflamm",
         pathway="Matrix stiffness", family="Other",
         genes=["COL15A1", "COL4A1", "CXCR4", "MMP7"],
         directions=[1, 1, 1, 1],
         source="survey_43_pool (row 3)",
         notes="All four matrix-stiffness biomarkers reported as upregulated (+1)."),
    dict(panel_id=104, pmid="41876560", year="2026", journal="NPJ Syst Biol Appl",
         pathway="Immune-metabolic crosstalk (gamma-delta T / endothelial)", family="Immune and inflammatory",
         genes=["ANXA6", "GLDC", "SLCO2B1", "TAPBPL", "TNFSF13", "GAS6"],
         directions=[0, 0, 0, 0, 0, 0],
         source="survey_43_pool (row 4)",
         notes="Six explicitly named symbols in text/tables evaluated verbatim (k=6)."),
    dict(panel_id=105, pmid="42776962", year="2026", journal="PLoS Comput Biol",
         pathway="Shared inflammatory genes (PD-sJIA)", family="Immune and inflammatory",
         genes=["CXCL1", "ELOVL4", "FAM46C", "LGALSL", "SELP", "VCAN"],
         directions=[0, 0, 0, 0, 0, 0],
         source="survey_43_pool (row 5)",
         notes="6-gene diagnostic panel from full-text tables; unsigned."),
    dict(panel_id=106, pmid="41358004", year="2025", journal="Front Cell Dev Biol",
         pathway="Immune / infective endocarditis comorbidity", family="Immune and inflammatory",
         genes=["ITGAM", "FCGR3B", "FCGR3A", "ITGB2"],
         directions=[1, 1, 1, 1],
         source="survey_43_pool (row 6)",
         notes="All four diagnostic markers reported as upregulated (+1)."),
    dict(panel_id=107, pmid="40539067", year="2025", journal="Front Immunol",
         pathway="Oxidative stress", family="Senescence and stress",
         genes=["COL4A2", "CYR61", "CXCL6"],
         directions=[0, 0, 0],
         source="survey_43_pool (row 7)",
         notes="Unsigned in summary; oriented by training-fold Welch t sign."),
    dict(panel_id=108, pmid="38218802", year="2024", journal="BMC Oral Health",
         pathway="Shared genes (PD-MS, complement / innate immunity)", family="Immune and inflammatory",
         genes=["FAM46C", "CFI", "DDIT4L"],
         directions=[0, 0, 0],
         source="survey_43_pool (row 8)",
         notes="Unsigned in summary; assigned to immune/complement family."),
    dict(panel_id=109, pmid="37051594", year="2023", journal="Front Genet",
         pathway="Shared genes (PD-T1D)", family="Metabolic",
         genes=["FMOD", "CD34", "TXN", "BBS7", "EGR1", "IGF2"],
         directions=[0, 0, 0, 0, 0, 0],
         source="overflow_8_pool (row 9)",
         notes="Group-level up/down counts stated without per-gene mapping; unsigned."),
    dict(panel_id=110, pmid="35145474", year="2021", journal="Front Endocrinol (Lausanne)",
         pathway="Shared genes (PD-T2D)", family="Metabolic",
         genes=["RAC2", "INPP5D", "PTPRC", "HGF"],
         directions=[1, 1, 1, 1],
         source="overflow_8_pool (row 10)",
         notes="All four hub genes reported as upregulated in both diseases (+1)."),
    dict(panel_id=111, pmid="39610070", year="2024", journal="Hua Xi Kou Qiang Yi Xue Za Zhi",
         pathway="Fatty acid metabolism", family="Metabolic",
         genes=["BTG2", "CXCL12", "FABP4", "CLDN10", "PPBP", "RGS1", "LGALSL", "RIF1"],
         directions=[0, 0, 0, 0, 0, 0, 0, 0],
         source="recovered_epmc_7_pool (row 1, AUC 0.967)",
         notes="8-gene LASSO+RFE+Boruta classifier on GSE10334 validated on GSE16134; unsigned."),
    dict(panel_id=112, pmid="41459503", year="2025", journal="Front Immunol",
         pathway="Integrated stress response (ISR)", family="Senescence and stress",
         genes=["BTG2", "DERL3", "FOS", "HSPA13", "YOD1"],
         directions=[1, 1, 1, 1, -1],
         source="overflow_8_pool (candidate 1)",
         notes="RT-qPCR verified BTG2/DERL3/FOS/HSPA13 upregulated (+1) and YOD1 downregulated (-1)."),
    dict(panel_id=113, pmid="39741754", year="2024", journal="J Inflamm Res",
         pathway="Mitochondrial dysfunction", family="Mitochondrial",
         genes=["BNIP3", "FAHD1", "UNG", "CBR3", "SLC25A43"],
         directions=[-1, -1, -1, 0, 0],
         source="overflow_8_pool (candidate 2)",
         notes="BNIP3/FAHD1/UNG reported as downregulated (-1); CBR3/SLC25A43 unsigned (0)."),
    dict(panel_id=114, pmid="40859550", year="2025", journal="Medicine (Baltimore)",
         pathway="Mitophagy", family="Mitochondrial",
         genes=["BNIP3L", "VPS13C", "CTTN", "MAP1LC3B"],
         directions=[-1, 1, -1, 1],
         source="overflow_8_pool (candidate 3)",
         notes="BNIP3L/CTTN downregulated (-1); VPS13C/MAP1LC3B upregulated (+1)."),
    dict(panel_id=115, pmid="40309038", year="2025", journal="Front Genet",
         pathway="Shared genes (SLE-PD)", family="Immune and inflammatory",
         genes=["PLEKHA1", "CEACAM1", "TNFAIP6", "TCN2", "GLDC", "GNG7",
                "LY96", "VCAN", "FCGR1B", "GZMK", "IFI27"],
         directions=[0] * 11,
         source="overflow_8_pool (Table 1, 11 hub genes)",
         notes="11 hub genes evaluated with individual and combined ROC AUC in Table 1; unsigned."),
]

ALIAS = {"ADGRF5": "GPR116"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--v3-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    g2p = json.load(open(os.path.join(args.data_dir, "gpl570_gene2probe.json")))
    exploration = json.load(open(os.path.join(args.data_dir, "panels.json")))
    assert len(exploration) == 27

    # ---- mapping
    out_panels = []
    rows = []
    for p in PANELS:
        genes = list(dict.fromkeys(p["genes"]))
        assert len(genes) == len(p["genes"]), f"duplicate gene in {p['panel_id']}"
        assert len(genes) == len(p["directions"])
        mapped = []
        missing = []
        for g, d in zip(genes, p["directions"]):
            sym = ALIAS.get(g, g)
            if sym not in g2p:
                missing.append(g)
                continue
            mapped.append({"gene": g, "symbol": sym, "probes": list(g2p[sym]),
                           "direction_published": int(d)})
        if len(mapped) < 3:
            raise SystemExit(f"panel {p['panel_id']} maps only {len(mapped)} genes")
        rec = dict(p)
        rec["genes"] = genes
        rec["mapped"] = mapped
        rec["n_genes"] = len(genes)
        rec["n_mapped"] = len(mapped)
        rec["n_direction_published"] = int(sum(1 for m in mapped if m["direction_published"] != 0))
        rec["missing_genes"] = missing
        out_panels.append(rec)
        rows.append({
            "panel_id": p["panel_id"], "pmid": p["pmid"], "year": p["year"],
            "journal": p["journal"], "pathway": p["pathway"], "family": p["family"],
            "genes": ";".join(genes), "n_genes": len(genes),
            "n_mapped": len(mapped), "mapped_symbols": ";".join(m["symbol"] for m in mapped),
            "n_direction_published": rec["n_direction_published"],
            "directions": ";".join(str(m["direction_published"]) for m in mapped),
            "missing_genes": ";".join(missing), "source": p["source"], "notes": p["notes"],
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out_dir, "01_panel_table.tsv"), sep="\t", index=False)
    json.dump(out_panels, open(os.path.join(args.out_dir, "01_panels_confirmation.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- pathway family table (fixed before fitting)
    fam_order = ["", "", "", "", "", ""]
    fam_rows = []
    for fam in fam_order:
        ids = [p["panel_id"] for p in out_panels if p["family"] == fam]
        fam_rows.append({"family": fam, "n_panels": len(ids),
                         "panel_ids": ";".join(str(i) for i in ids),
                         "n_genes": int(sum(p["n_mapped"] for p in out_panels if p["family"] == fam))})
    pd.DataFrame(fam_rows).to_csv(os.path.join(args.out_dir, "02_pathway_families.tsv"),
                                  sep="\t", index=False)

    # ---- fixed shared-axis reference: 27 exploration panels only
    rec = {}
    for p in exploration:
        for m in p["mapped"]:
            r = rec.setdefault(m["symbol"], {"directions": [], "panels": set(), "originals": set()})
            r["panels"].add(p["panel_id"])
            r["originals"].add(m["gene"])
            d = m.get("direction_published", 0)
            if d != 0:
                r["directions"].append(d)
    ref_rows = []
    for sym in sorted(rec):
        dirs = rec[sym]["directions"]
        if not dirs:
            d = 0
        elif sum(dirs) > 0:
            d = 1
        elif sum(dirs) < 0:
            d = -1
        else:
            d = 0
        ref_rows.append({"gene": sym, "direction": d,
                         "n_panels": len(rec[sym]["panels"]),
                         "source_panel_ids": ";".join(str(i) for i in sorted(rec[sym]["panels"])),
                         "original_symbols": ";".join(sorted(rec[sym]["originals"])),
                         "n_direction_published": len(dirs)})
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(os.path.join(args.out_dir, "03_shared_axis_reference.tsv"), sep="\t", index=False)
    assert len(ref) == 133, len(ref)

    # ---- provenance hashes of inputs
    sources = [
        os.path.join(args.out_dir, "..", "_v0.md"),

        os.path.join(args.v3_dir, "01_protocol_v3.md"),
        os.path.join(args.v3_dir, "scripts", "common.py"),
        os.path.join(args.v3_dir, "scripts", "03_random_panels.py"),
        os.path.join(args.v3_dir, "scripts", "04_incremental_v3.py"),
        os.path.join(args.v3_dir, "scripts", "07_selection_bootstrap.py"),
        os.path.join(args.data_dir, "gse16134.npz"),
        os.path.join(args.data_dir, "panels.json"),
        os.path.join(args.data_dir, "gpl570_gene2probe.json"),
        os.path.join(args.v3_dir, "results", "markers_celltype_top_v3.csv"),
        os.path.join(args.v3_dir, "results", "markers_williams_derived_top.csv"),
        os.path.join(args.v3_dir, "results", "markers_williams_published.csv"),
        os.path.join(args.v3_dir, "results", "gene_t_full.csv"),
        os.path.join(args.v3_dir, "results", "matching_bins.json"),
    ]
    prov = {}
    for s in sources:
        if os.path.exists(s):
            prov[os.path.relpath(os.path.abspath(s), os.path.abspath(args.out_dir))] = sha256(s)
        else:
            prov[os.path.relpath(os.path.abspath(s), os.path.abspath(args.out_dir))] = "MISSING"
    json.dump(prov, open(os.path.join(args.out_dir, "provenance_inputs.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- frozen hashes of the pre-model artefacts
    hash_targets = ["00_protocol_confirmation.md", "01_panel_table.tsv",
                    "01_panels_confirmation.json", "02_pathway_families.tsv",
                    "03_shared_axis_reference.tsv"]
    lines = []
    for rel in hash_targets:
        p = os.path.join(args.out_dir, rel)
        lines.append(f"{sha256(p)}  {rel}")
    open(os.path.join(args.out_dir, "SHA256SUMS_premodel.txt"), "w").write("\n".join(lines) + "\n")

    print(tab[["panel_id", "pmid", "family", "n_genes", "n_mapped", "n_direction_published"]]
          .to_string(index=False))
    print("\nfamily counts:")
    print(pd.DataFrame(fam_rows).to_string(index=False))
    print(f"\nshared reference: {len(ref)} genes; hashes written to SHA256SUMS_premodel.txt")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    main()
