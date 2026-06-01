"""
drug_target_db.py - Curated drug-target-modality mappings and biomarker patterns.

This is a lookup table. In production, you'd replace/supplement this with
live queries to ChEMBL, DrugBank, Open Targets, etc.
"""

# ============================================================
# DRUG -> TARGET / MODALITY / MOA DATABASE
# ============================================================
# Format: "drug_name_lowercase": {
#     "targets": [list of HGNC gene symbols],
#     "modality": "small_molecule" | "antibody" | "adc" | "bispecific" | "cell_therapy",
#     "moa": "mechanism_of_action_tag"
# }

DRUG_TARGET_DB = {
    # --- Checkpoint inhibitors ---
    "pembrolizumab":  {"targets": ["PDCD1"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "keytruda":       {"targets": ["PDCD1"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "nivolumab":      {"targets": ["PDCD1"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "opdivo":         {"targets": ["PDCD1"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "atezolizumab":   {"targets": ["CD274"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "tecentriq":      {"targets": ["CD274"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "durvalumab":     {"targets": ["CD274"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "imfinzi":        {"targets": ["CD274"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "avelumab":       {"targets": ["CD274"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "ipilimumab":     {"targets": ["CTLA4"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "yervoy":         {"targets": ["CTLA4"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},
    "tremelimumab":   {"targets": ["CTLA4"],  "modality": "antibody", "moa": "checkpoint_inhibitor"},

    # --- PARP inhibitors ---
    "olaparib":       {"targets": ["PARP1", "PARP2"], "modality": "small_molecule", "moa": "parp_inhibitor"},
    "lynparza":       {"targets": ["PARP1", "PARP2"], "modality": "small_molecule", "moa": "parp_inhibitor"},
    "rucaparib":      {"targets": ["PARP1"],          "modality": "small_molecule", "moa": "parp_inhibitor"},
    "niraparib":      {"targets": ["PARP1", "PARP2"], "modality": "small_molecule", "moa": "parp_inhibitor"},
    "talazoparib":    {"targets": ["PARP1"],          "modality": "small_molecule", "moa": "parp_inhibitor"},

    # --- KRAS ---
    "sotorasib":      {"targets": ["KRAS"], "modality": "small_molecule", "moa": "kras_inhibitor"},
    "adagrasib":      {"targets": ["KRAS"], "modality": "small_molecule", "moa": "kras_inhibitor"},

    # --- EGFR ---
    "osimertinib":    {"targets": ["EGFR"], "modality": "small_molecule", "moa": "egfr_inhibitor"},
    "tagrisso":       {"targets": ["EGFR"], "modality": "small_molecule", "moa": "egfr_inhibitor"},
    "erlotinib":      {"targets": ["EGFR"], "modality": "small_molecule", "moa": "egfr_inhibitor"},
    "gefitinib":      {"targets": ["EGFR"], "modality": "small_molecule", "moa": "egfr_inhibitor"},
    "afatinib":       {"targets": ["EGFR"], "modality": "small_molecule", "moa": "egfr_inhibitor"},
    "cetuximab":      {"targets": ["EGFR"], "modality": "antibody",      "moa": "egfr_targeting"},
    "panitumumab":    {"targets": ["EGFR"], "modality": "antibody",      "moa": "egfr_targeting"},

    # --- BRAF / MEK ---
    "vemurafenib":    {"targets": ["BRAF"],   "modality": "small_molecule", "moa": "braf_inhibitor"},
    "dabrafenib":     {"targets": ["BRAF"],   "modality": "small_molecule", "moa": "braf_inhibitor"},
    "encorafenib":    {"targets": ["BRAF"],   "modality": "small_molecule", "moa": "braf_inhibitor"},
    "trametinib":     {"targets": ["MAP2K1"], "modality": "small_molecule", "moa": "mek_inhibitor"},
    "cobimetinib":    {"targets": ["MAP2K1"], "modality": "small_molecule", "moa": "mek_inhibitor"},
    "binimetinib":    {"targets": ["MAP2K1"], "modality": "small_molecule", "moa": "mek_inhibitor"},

    # --- ALK ---
    "crizotinib":     {"targets": ["ALK", "MET"], "modality": "small_molecule", "moa": "alk_inhibitor"},
    "alectinib":      {"targets": ["ALK"],        "modality": "small_molecule", "moa": "alk_inhibitor"},
    "lorlatinib":     {"targets": ["ALK"],        "modality": "small_molecule", "moa": "alk_inhibitor"},
    "ceritinib":      {"targets": ["ALK"],        "modality": "small_molecule", "moa": "alk_inhibitor"},
    "brigatinib":     {"targets": ["ALK"],        "modality": "small_molecule", "moa": "alk_inhibitor"},

    # --- HER2 ---
    "trastuzumab":              {"targets": ["ERBB2"],         "modality": "antibody",       "moa": "her2_targeting"},
    "herceptin":                {"targets": ["ERBB2"],         "modality": "antibody",       "moa": "her2_targeting"},
    "pertuzumab":               {"targets": ["ERBB2"],         "modality": "antibody",       "moa": "her2_targeting"},
    "trastuzumab deruxtecan":   {"targets": ["ERBB2"],         "modality": "adc",            "moa": "her2_adc"},
    "t-dxd":                    {"targets": ["ERBB2"],         "modality": "adc",            "moa": "her2_adc"},
    "enhertu":                  {"targets": ["ERBB2"],         "modality": "adc",            "moa": "her2_adc"},
    "ado-trastuzumab emtansine":{"targets": ["ERBB2"],         "modality": "adc",            "moa": "her2_adc"},
    "lapatinib":                {"targets": ["ERBB2", "EGFR"], "modality": "small_molecule", "moa": "her2_inhibitor"},
    "neratinib":                {"targets": ["ERBB2"],         "modality": "small_molecule", "moa": "her2_inhibitor"},
    "tucatinib":                {"targets": ["ERBB2"],         "modality": "small_molecule", "moa": "her2_inhibitor"},

    # --- CDK4/6 ---
    "palbociclib":    {"targets": ["CDK4", "CDK6"], "modality": "small_molecule", "moa": "cdk_inhibitor"},
    "ribociclib":     {"targets": ["CDK4", "CDK6"], "modality": "small_molecule", "moa": "cdk_inhibitor"},
    "abemaciclib":    {"targets": ["CDK4", "CDK6"], "modality": "small_molecule", "moa": "cdk_inhibitor"},

    # --- BTK ---
    "ibrutinib":      {"targets": ["BTK"], "modality": "small_molecule", "moa": "btk_inhibitor"},
    "acalabrutinib":  {"targets": ["BTK"], "modality": "small_molecule", "moa": "btk_inhibitor"},
    "zanubrutinib":   {"targets": ["BTK"], "modality": "small_molecule", "moa": "btk_inhibitor"},

    # --- BCL2 ---
    "venetoclax":     {"targets": ["BCL2"], "modality": "small_molecule", "moa": "bcl2_inhibitor"},

    # --- Multikinase ---
    "sorafenib":      {"targets": ["BRAF", "VEGFR2"],       "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "sunitinib":      {"targets": ["VEGFR2", "PDGFRA", "KIT"], "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "lenvatinib":     {"targets": ["VEGFR2", "FGFR1"],      "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "cabozantinib":   {"targets": ["MET", "VEGFR2"],         "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "regorafenib":    {"targets": ["VEGFR2", "KIT"],         "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "axitinib":       {"targets": ["VEGFR2"],                "modality": "small_molecule", "moa": "vegfr_inhibitor"},
    "pazopanib":      {"targets": ["VEGFR2", "PDGFRA"],      "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "imatinib":       {"targets": ["ABL1", "KIT"],           "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "dasatinib":      {"targets": ["ABL1", "SRC"],           "modality": "small_molecule", "moa": "multikinase_inhibitor"},
    "nilotinib":      {"targets": ["ABL1"],                  "modality": "small_molecule", "moa": "abl_inhibitor"},

    # --- VEGF antibodies ---
    "bevacizumab":    {"targets": ["VEGFA"], "modality": "antibody", "moa": "vegf_targeting"},
    "avastin":        {"targets": ["VEGFA"], "modality": "antibody", "moa": "vegf_targeting"},
    "ramucirumab":    {"targets": ["KDR"],   "modality": "antibody", "moa": "vegfr_targeting"},

    # --- CD20 ---
    "rituximab":      {"targets": ["MS4A1"], "modality": "antibody", "moa": "cd20_targeting"},
    "obinutuzumab":   {"targets": ["MS4A1"], "modality": "antibody", "moa": "cd20_targeting"},

    # --- mTOR ---
    "everolimus":     {"targets": ["MTOR"], "modality": "small_molecule", "moa": "mtor_inhibitor"},
    "temsirolimus":   {"targets": ["MTOR"], "modality": "small_molecule", "moa": "mtor_inhibitor"},

    # --- PI3K ---
    "alpelisib":      {"targets": ["PIK3CA"], "modality": "small_molecule", "moa": "pi3k_inhibitor"},
    "idelalisib":     {"targets": ["PIK3CD"], "modality": "small_molecule", "moa": "pi3k_inhibitor"},

    # --- NTRK / ROS1 ---
    "larotrectinib":  {"targets": ["NTRK1", "NTRK2", "NTRK3"], "modality": "small_molecule", "moa": "trk_inhibitor"},
    "entrectinib":    {"targets": ["NTRK1", "ROS1", "ALK"],    "modality": "small_molecule", "moa": "trk_inhibitor"},

    # --- ADCs ---
    "sacituzumab govitecan": {"targets": ["TACSTD2"],  "modality": "adc", "moa": "trop2_adc"},
    "enfortumab vedotin":    {"targets": ["NECTIN4"],  "modality": "adc", "moa": "nectin4_adc"},
    "brentuximab vedotin":   {"targets": ["TNFRSF8"],  "modality": "adc", "moa": "cd30_adc"},

    # --- Hormonal ---
    "enzalutamide":   {"targets": ["AR"],      "modality": "small_molecule", "moa": "ar_inhibitor"},
    "abiraterone":    {"targets": ["CYP17A1"], "modality": "small_molecule", "moa": "cyp17_inhibitor"},
    "tamoxifen":      {"targets": ["ESR1"],    "modality": "small_molecule", "moa": "serm"},
    "letrozole":      {"targets": ["CYP19A1"], "modality": "small_molecule", "moa": "aromatase_inhibitor"},
    "fulvestrant":    {"targets": ["ESR1"],    "modality": "small_molecule", "moa": "serd"},

    # --- Other targeted ---
    "bortezomib":     {"targets": ["PSMB5"], "modality": "small_molecule", "moa": "proteasome_inhibitor"},
    "carfilzomib":    {"targets": ["PSMB5"], "modality": "small_molecule", "moa": "proteasome_inhibitor"},
    "lenalidomide":   {"targets": ["CRBN"],  "modality": "small_molecule", "moa": "imid"},
    "pomalidomide":   {"targets": ["CRBN"],  "modality": "small_molecule", "moa": "imid"},
    "thalidomide":    {"targets": ["CRBN"],  "modality": "small_molecule", "moa": "imid"},
    "vismodegib":     {"targets": ["SMO"],   "modality": "small_molecule", "moa": "hedgehog_inhibitor"},
    "ruxolitinib":    {"targets": ["JAK1", "JAK2"], "modality": "small_molecule", "moa": "jak_inhibitor"},
    "tofacitinib":    {"targets": ["JAK1", "JAK3"], "modality": "small_molecule", "moa": "jak_inhibitor"},

    # --- Common chemo (lower mechanistic specificity) ---
    "capecitabine":     {"targets": ["TYMS"],          "modality": "small_molecule", "moa": "antimetabolite"},
    "gemcitabine":      {"targets": ["RRM1"],          "modality": "small_molecule", "moa": "antimetabolite"},
    "cisplatin":        {"targets": ["DNA"],           "modality": "small_molecule", "moa": "dna_crosslinker"},
    "carboplatin":      {"targets": ["DNA"],           "modality": "small_molecule", "moa": "dna_crosslinker"},
    "oxaliplatin":      {"targets": ["DNA"],           "modality": "small_molecule", "moa": "dna_crosslinker"},
    "paclitaxel":       {"targets": ["TUBB"],          "modality": "small_molecule", "moa": "tubulin_inhibitor"},
    "docetaxel":        {"targets": ["TUBB"],          "modality": "small_molecule", "moa": "tubulin_inhibitor"},
    "doxorubicin":      {"targets": ["TOP2A"],         "modality": "small_molecule", "moa": "topoisomerase_inhibitor"},
    "irinotecan":       {"targets": ["TOP1"],          "modality": "small_molecule", "moa": "topoisomerase_inhibitor"},
    "topotecan":        {"targets": ["TOP1"],          "modality": "small_molecule", "moa": "topoisomerase_inhibitor"},
    "etoposide":        {"targets": ["TOP2A"],         "modality": "small_molecule", "moa": "topoisomerase_inhibitor"},
    "temozolomide":     {"targets": ["DNA"],           "modality": "small_molecule", "moa": "alkylating_agent"},
    "cyclophosphamide": {"targets": ["DNA"],           "modality": "small_molecule", "moa": "alkylating_agent"},
    "fluorouracil":     {"targets": ["TYMS"],          "modality": "small_molecule", "moa": "antimetabolite"},
    "5-fu":             {"targets": ["TYMS"],          "modality": "small_molecule", "moa": "antimetabolite"},
    "pemetrexed":       {"targets": ["TYMS", "DHFR"],  "modality": "small_molecule", "moa": "antimetabolite"},
    "methotrexate":     {"targets": ["DHFR"],          "modality": "small_molecule", "moa": "antimetabolite"},
}


# ============================================================
# BIOMARKER PATTERNS (for text extraction from trial titles/eligibility)
# ============================================================
BIOMARKER_PATTERNS = {
    # Immune checkpoint biomarkers
    "PD-L1":              ["pd-l1", "pdl1", "pd-l1 positive", "pd-l1 expression", "cps", "tps", "combined positive score"],
    "TMB-H":              ["tmb-h", "tmb high", "tumor mutational burden", "tmb", "high tmb"],
    "MSI-H/dMMR":         ["msi-h", "msi high", "dmmr", "microsatellite instability", "mismatch repair", "mmr deficient"],
    # Specific mutations
    "BRCA1/2":            ["brca", "brca1", "brca2", "brca mutation", "brca-mutated", "brca positive", "hrd", "homologous recombination"],
    "KRAS G12C":          ["kras g12c", "kras-g12c"],
    "KRAS":               ["kras mutation", "kras mutant", "kras-mutated", "kras positive"],
    "EGFR mutation":      ["egfr mutation", "egfr mutant", "egfr-positive", "egfr+", "egfr t790m", "egfr exon", "egfr del19", "egfr l858r", "egfr-mutated"],
    "ALK fusion":         ["alk fusion", "alk-positive", "alk rearrangement", "alk+", "alk-rearranged", "eml4-alk"],
    "BRAF V600E":         ["braf v600", "braf mutation", "braf-mutant", "braf-mutated", "braf positive"],
    "HER2 amplification": ["her2-positive", "her2+", "her2 amplif", "erbb2", "her2-overexpressing", "her2 positive", "her2-amplified"],
    "NTRK fusion":        ["ntrk fusion", "ntrk rearrangement", "ntrk-positive"],
    "ROS1 fusion":        ["ros1 fusion", "ros1 rearrangement", "ros1-positive", "ros1+"],
    "PIK3CA mutation":    ["pik3ca", "pi3k mutation", "pik3ca-mutated"],
    "FGFR alteration":    ["fgfr alteration", "fgfr fusion", "fgfr mutation", "fgfr2", "fgfr3", "fgfr-altered"],
    "MET amplification":  ["met amplification", "met exon 14", "c-met", "met-positive", "met overexpression"],
    "TP53 mutation":      ["tp53", "p53 mutation", "tp53-mutated"],
    "PTEN loss":          ["pten loss", "pten deletion", "pten-deficient"],
    "BCL2 overexpression":["bcl2", "bcl-2 overexpression", "bcl-2 positive"],
    "IDH1 mutation":      ["idh1 mutation", "idh1 mutant", "idh1-mutated", "idh mutation"],
    "IDH2 mutation":      ["idh2 mutation", "idh2 mutant"],
    "RET fusion":         ["ret fusion", "ret rearrangement", "ret-positive", "ret mutation"],
    "ERBB2 mutation":     ["erbb2 mutation", "her2 mutation", "her2-mutated"],
    # Hormone receptors
    "ER positive":        ["er-positive", "er+", "estrogen receptor positive", "hormone receptor positive", "hr+", "hr-positive"],
    "ER negative":        ["er-negative", "er-", "estrogen receptor negative"],
    "PR positive":        ["pr-positive", "pr+", "progesterone receptor positive"],
    "AR positive":        ["androgen receptor", "ar-positive", "ar+"],
    "Triple negative":    ["triple-negative", "triple negative", "tnbc"],
    # Other genomic
    "FLT3 mutation":      ["flt3 mutation", "flt3-itd", "flt3 itd"],
    "NPM1 mutation":      ["npm1 mutation", "npm1-mutated"],
    "ARID1A loss":        ["arid1a", "arid1a loss", "arid1a mutation"],
    "ATM loss":           ["atm loss", "atm mutation", "atm-deficient"],
    "CDH1 mutation":      ["cdh1 mutation", "e-cadherin"],
    "PALB2 mutation":     ["palb2 mutation", "palb2"],
    # Expression biomarkers
    "Trop-2":             ["trop-2", "trop2", "tacstd2"],
    "Nectin-4":           ["nectin-4", "nectin4"],
    "PSMA":               ["psma", "prostate-specific membrane antigen"],
    "CD19":               ["cd19", "cd19-positive", "cd19+"],
    "CD20":               ["cd20", "cd20-positive", "cd20+"],
    "CD30":               ["cd30", "cd30-positive", "cd30+"],
    "CD38":               ["cd38", "cd38-positive", "cd38+"],
    # General genomic
    "Biomarker selected":  ["biomarker-selected", "biomarker selected", "molecularly selected",
                            "genomically selected", "mutation-selected", "genotype-selected",
                            "biomarker-driven", "precision", "targeted therapy based on"],
}


# ============================================================
# TARGET-LEVEL PROPERTIES (what you'd get from Open Targets, DepMap, etc.)
# In production, replace with live API calls.
# Format: target -> (ot_assoc, genetics, somatic, dependency, degree, druggability, pub_count)
# ============================================================
TARGET_PROPERTIES = {
    "EGFR":    (0.85, 0.80, 0.90, 0.75, 180, 0.95, 15000),
    "BRAF":    (0.82, 0.70, 0.95, 0.80, 120, 0.90, 8000),
    "ERBB2":   (0.88, 0.60, 0.85, 0.70, 150, 0.92, 12000),
    "ALK":     (0.78, 0.55, 0.80, 0.72, 80,  0.88, 5000),
    "KRAS":    (0.90, 0.85, 0.95, 0.85, 200, 0.70, 18000),
    "PDCD1":   (0.70, 0.30, 0.20, 0.30, 50,  0.85, 20000),
    "CD274":   (0.68, 0.25, 0.25, 0.25, 45,  0.82, 18000),
    "CTLA4":   (0.55, 0.20, 0.15, 0.20, 40,  0.80, 10000),
    "PARP1":   (0.75, 0.50, 0.60, 0.65, 250, 0.88, 7000),
    "PARP2":   (0.60, 0.40, 0.45, 0.55, 100, 0.85, 2000),
    "BTK":     (0.80, 0.55, 0.40, 0.75, 70,  0.90, 4000),
    "BCL2":    (0.78, 0.45, 0.50, 0.70, 110, 0.85, 9000),
    "ABL1":    (0.82, 0.65, 0.70, 0.60, 130, 0.92, 6000),
    "KIT":     (0.70, 0.55, 0.75, 0.55, 90,  0.85, 5000),
    "CDK4":    (0.72, 0.40, 0.50, 0.60, 100, 0.88, 4000),
    "CDK6":    (0.68, 0.35, 0.45, 0.55, 95,  0.85, 3000),
    "MAP2K1":  (0.65, 0.30, 0.55, 0.50, 85,  0.82, 3500),
    "MTOR":    (0.60, 0.25, 0.35, 0.45, 300, 0.80, 12000),
    "MET":     (0.72, 0.50, 0.70, 0.60, 110, 0.85, 6000),
    "VEGFA":   (0.50, 0.15, 0.10, 0.20, 120, 0.75, 15000),
    "VEGFR2":  (0.55, 0.20, 0.15, 0.25, 80,  0.80, 8000),
    "KDR":     (0.52, 0.18, 0.12, 0.22, 75,  0.78, 5000),
    "MS4A1":   (0.65, 0.10, 0.15, 0.40, 30,  0.90, 4000),
    "NTRK1":   (0.70, 0.60, 0.65, 0.68, 60,  0.82, 2000),
    "NTRK2":   (0.55, 0.45, 0.50, 0.50, 55,  0.75, 1500),
    "NTRK3":   (0.55, 0.45, 0.50, 0.50, 50,  0.75, 1500),
    "ROS1":    (0.60, 0.50, 0.55, 0.55, 40,  0.80, 2500),
    "FGFR1":   (0.58, 0.40, 0.50, 0.45, 90,  0.78, 4000),
    "PDGFRA":  (0.55, 0.35, 0.50, 0.40, 70,  0.80, 3500),
    "TACSTD2": (0.50, 0.15, 0.30, 0.35, 25,  0.70, 800),
    "NECTIN4": (0.48, 0.10, 0.25, 0.30, 20,  0.65, 500),
    "TNFRSF8": (0.55, 0.15, 0.20, 0.35, 30,  0.72, 2000),
    "AR":      (0.80, 0.60, 0.40, 0.65, 200, 0.90, 14000),
    "CYP17A1": (0.65, 0.30, 0.20, 0.40, 50,  0.85, 3000),
    "ESR1":    (0.75, 0.50, 0.30, 0.55, 150, 0.88, 12000),
    "CYP19A1": (0.70, 0.45, 0.25, 0.50, 60,  0.85, 5000),
    "PIK3CA":  (0.72, 0.55, 0.65, 0.50, 130, 0.78, 6000),
    "PIK3CD":  (0.60, 0.40, 0.35, 0.45, 80,  0.75, 2500),
    "PSMB5":   (0.65, 0.20, 0.15, 0.55, 40,  0.82, 3000),
    "CRBN":    (0.60, 0.15, 0.10, 0.50, 35,  0.78, 2500),
    "SMO":     (0.55, 0.40, 0.30, 0.40, 45,  0.75, 2000),
    "JAK1":    (0.62, 0.35, 0.25, 0.50, 100, 0.82, 5000),
    "JAK2":    (0.65, 0.45, 0.35, 0.55, 90,  0.80, 6000),
    "SRC":     (0.50, 0.20, 0.30, 0.35, 200, 0.75, 8000),
    # Chemo targets
    "TYMS":    (0.40, 0.15, 0.10, 0.30, 40,  0.70, 4000),
    "RRM1":    (0.38, 0.12, 0.10, 0.28, 35,  0.65, 2000),
    "DNA":     (0.30, 0.05, 0.05, 0.20, 500, 0.60, 20000),
    "TUBB":    (0.35, 0.10, 0.08, 0.25, 60,  0.65, 5000),
    "TOP2A":   (0.42, 0.15, 0.12, 0.35, 80,  0.70, 4000),
    "TOP1":    (0.40, 0.12, 0.10, 0.32, 50,  0.68, 3000),
    "DHFR":    (0.35, 0.10, 0.08, 0.25, 45,  0.65, 3000),
}

DEFAULT_TARGET_PROPS = (0.25, 0.10, 0.10, 0.15, 20, 0.30, 100)


# ============================================================
# BIOLOGY PAIR STRENGTH (target-disease combos for label heuristics)
# ============================================================
STRONG_PAIRS = {
    ("EGFR", "NSCLC"), ("ALK", "NSCLC"), ("BRAF", "melanoma"),
    ("ERBB2", "HER2+ breast cancer"), ("ERBB2", "breast cancer"),
    ("PDCD1", "melanoma"), ("PDCD1", "NSCLC"), ("PDCD1", "bladder cancer"),
    ("CD274", "NSCLC"), ("CD274", "bladder cancer"),
    ("PARP1", "ovarian cancer"), ("PARP2", "ovarian cancer"),
    ("PARP1", "breast cancer"), ("PARP1", "TNBC"),
    ("BTK", "CLL"), ("BTK", "DLBCL"),
    ("BCL2", "CLL"), ("BCL2", "AML"),
    ("CDK4", "breast cancer"), ("CDK6", "breast cancer"),
    ("MS4A1", "DLBCL"), ("MS4A1", "lymphoma"),
    ("KRAS", "NSCLC"), ("KRAS", "colorectal cancer"),
    ("NTRK1", "thyroid cancer"), ("NTRK1", "solid tumors"),
    ("MAP2K1", "melanoma"), ("MTOR", "renal cell carcinoma"),
    ("CTLA4", "melanoma"), ("VEGFA", "colorectal cancer"),
    ("VEGFA", "renal cell carcinoma"), ("TACSTD2", "TNBC"),
    ("NECTIN4", "bladder cancer"), ("AR", "prostate cancer"),
    ("CYP17A1", "prostate cancer"), ("PSMB5", "multiple myeloma"),
    ("CRBN", "multiple myeloma"), ("ABL1", "CLL"),
    ("ESR1", "breast cancer"), ("CYP19A1", "breast cancer"),
    ("PIK3CA", "breast cancer"),
}

MEDIUM_PAIRS = {
    ("PDCD1", "HNSCC"), ("PDCD1", "renal cell carcinoma"),
    ("PDCD1", "gastric cancer"), ("PDCD1", "hepatocellular carcinoma"),
    ("CD274", "renal cell carcinoma"), ("CD274", "TNBC"),
    ("EGFR", "HNSCC"), ("EGFR", "colorectal cancer"),
    ("BRAF", "colorectal cancer"), ("BRAF", "thyroid cancer"),
    ("MET", "NSCLC"), ("MET", "gastric cancer"),
    ("VEGFR2", "hepatocellular carcinoma"), ("VEGFR2", "renal cell carcinoma"),
    ("PARP1", "prostate cancer"), ("PARP1", "pancreatic cancer"),
    ("CDK4", "endometrial cancer"), ("MTOR", "breast cancer"),
    ("FGFR1", "cholangiocarcinoma"), ("VEGFA", "gastric cancer"),
    ("JAK1", "lymphoma"), ("JAK2", "AML"),
    ("TOP1", "colorectal cancer"), ("TUBB", "NSCLC"),
    ("TUBB", "breast cancer"), ("TYMS", "colorectal cancer"),
    ("DNA", "ovarian cancer"), ("DNA", "SCLC"),
    ("RRM1", "NSCLC"), ("RRM1", "pancreatic cancer"),
    ("TOP2A", "AML"), ("TOP2A", "breast cancer"),
    ("DHFR", "lymphoma"),
}

print("drug_target_db.py loaded successfully.")