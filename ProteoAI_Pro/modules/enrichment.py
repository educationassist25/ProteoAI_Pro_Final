"""
enrichment.py - Gene Set Enrichment Analysis for proteomics, offering three
statistically-distinct methods:

1. STRING Enrichment Analysis (Szklarczyk D et al. 2023, "The STRING database in
   2023", Nucleic Acids Research) -- hypergeometric-style functional enrichment
   computed SERVER-SIDE by STRING against STRING's own bundled/versioned
   GO/KEGG/Reactome/Pfam/InterPro annotation sets. Needs only a protein list;
   supports every organism in STRING's database (thousands of species via NCBI
   taxonomy ID), including plants. STRING's own database version (queried live via
   `get_string_version()`) governs the annotation vintage here -- NOT Enrichr's
   dated GO/KEGG library names below, which are a separate resource.

2. Over-Representation Analysis (ORA) -- a hypergeometric test computed LOCALLY
   (not via Enrichr's own /enrich endpoint, which does not expose the underlying
   universe/gene-set/comparison/overlap counts this app reports). Gene sets are
   fetched in GMT format from the appropriate Enrichr-family instance for the
   selected organism (see ORGANISM_CATALOG) and tested one at a time:
       N = |universe|, K = |gene set ∩ universe|,
       n = |comparison ∩ universe|, k = |comparison ∩ gene set ∩ universe|
       P-value = P(X >= k) via the hypergeometric survival function
       Fold Enrichment = (k/n) / (K/N)
   BH-FDR is applied across every gene set actually tested. The FULL
   comparison/input gene list is used as given -- no additional cutoff is applied
   inside this module.

3. Gene Set Enrichment Analysis, GSEA (Subramanian A et al. 2005, PNAS "Gene set
   enrichment analysis: a knowledge-based approach..."; gene-set permutation as
   implemented by GSEAPY's prerank mode, Fang Z et al. 2023, Bioinformatics) --
   ranks ALL detected proteins by a signed score and computes a weighted
   running-sum enrichment statistic (ES, normalized to NES) per gene set, with
   permutation-based significance and reported leading-edge genes. This is a
   textbook-distinct algorithm from ORA/the hypergeometric test above -- it is
   never used as a substitute for Method 2, and vice versa.

All methods expect gene SYMBOLS (not raw UniProt accessions) as protein
identifiers -- the convention Enrichr, STRING, and MSigDB/.gmt gene sets all
share. If your protein intensity matrix uses a different identifier, convert to
gene symbol (e.g. via UniProt's ID mapping service, or the "Gene names" column
many search engines like MaxQuant already report) before using this tab.

Database versions (verified against the live Enrichr library list; see
ORGANISM_CATALOG for the authoritative, per-organism, per-collection record --
never assume a "2026" label without verifying):
  - GO Biological/Molecular Function/Cellular Component: 2026 releases genuinely
    exist on Enrichr for HUMAN (GO_Biological_Process_2026, etc.) as of this
    writing. No organism-specific 2026 GO release exists for mouse/yeast/
    zebrafish on Enrichr; those use whatever that organism's own portal provides.
  - KEGG: a 2026 release genuinely exists for human (KEGG_2026, no species
    suffix in Enrichr's newer naming). The most recent mouse-specific KEGG
    library on Enrichr is KEGG_2019_Mouse -- explicitly NOT relabeled 2026.
  - Reactome: the most recent Enrichr release is Reactome_Pathways_2024 -- there
    is no 2026 Reactome library on Enrichr as of this writing.
  - WikiPathways: the most recent Enrichr releases are WikiPathways_2024_Human
    and WikiPathways_2024_Mouse -- no 2026 release exists.
  - MSigDB: the most recent Enrichr release is MSigDB_Hallmark_2020 -- no newer
    MSigDB library is available via Enrichr as of this writing.
"""

import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
YEAST_ENRICHR_BASE = "https://maayanlab.cloud/YeastEnrichr"
FISH_ENRICHR_BASE = "https://maayanlab.cloud/FishEnrichr"
STRING_BASE = "https://string-db.org/api"


class EnrichmentError(Exception):
    pass


# ---------------------------------------------------------------------------
# Organism / database catalog
#
# Every entry below was checked against the live Enrichr library list (fetched
# through a deployed third-party tool that queries Enrichr's own
# `datasetStatistics` endpoint) rather than assumed. "library" is an exact,
# confirmed-real Enrichr library name. "library_prefix" (used only for the
# organism-specific Enrichr instances, where Enrichr does not publish a
# year-stamped name) is resolved against that instance's live library list at
# query time via resolve_library_name() -- whatever name is actually resolved
# is recorded in the run metadata for reproducibility. "available=False"
# entries are NOT silently substituted with a different database; the UI
# disables them and shows the "note" explaining why, and points to STRING
# Enrichment Analysis instead where that covers the organism.
# ---------------------------------------------------------------------------
ORGANISM_CATALOG = {
    "Human (Homo sapiens)": {
        "string_taxid": 9606,
        "enrichr_base": ENRICHR_BASE,
        "gene_id_type": "Gene Symbol (HGNC)",
        "databases": {
            "GO Biological Process": {"library": "GO_Biological_Process_2026", "version": "2026", "available": True},
            "GO Molecular Function": {"library": "GO_Molecular_Function_2026", "version": "2026", "available": True},
            "GO Cellular Component": {"library": "GO_Cellular_Component_2026", "version": "2026", "available": True},
            "KEGG": {"library": "KEGG_2026", "version": "2026", "available": True},
            "Reactome": {"library": "Reactome_Pathways_2024", "version": "2024 (most recent Enrichr release; no 2026 Reactome library exists as of this writing)", "available": True},
            "WikiPathways": {"library": "WikiPathways_2024_Human", "version": "2024 (most recent Enrichr release; no 2026 release exists)", "available": True},
            "MSigDB (Hallmark)": {"library": "MSigDB_Hallmark_2020", "version": "2020 Hallmark collection (most recent MSigDB library on Enrichr; no newer release exists)", "available": True},
        },
    },
    "Mouse (Mus musculus)": {
        "string_taxid": 10090,
        "enrichr_base": ENRICHR_BASE,
        "gene_id_type": "Gene Symbol (MGI)",
        "databases": {
            "GO Biological Process": {"library": None, "version": None, "available": False,
                                       "note": "Enrichr's GO libraries are human-gene-symbol only; no mouse-specific GO release exists on Enrichr. Use STRING Enrichment Analysis for mouse GO-style annotation."},
            "GO Molecular Function": {"library": None, "version": None, "available": False,
                                       "note": "Enrichr's GO libraries are human-gene-symbol only. Use STRING Enrichment Analysis for mouse."},
            "GO Cellular Component": {"library": None, "version": None, "available": False,
                                       "note": "Enrichr's GO libraries are human-gene-symbol only. Use STRING Enrichment Analysis for mouse."},
            "KEGG": {"library": "KEGG_2019_Mouse", "version": "2019 (most recent mouse-specific KEGG library on Enrichr; no 2026 release exists)", "available": True},
            "Reactome": {"library": None, "version": None, "available": False,
                         "note": "No mouse-specific Reactome library exists on Enrichr. Use STRING Enrichment Analysis for mouse."},
            "WikiPathways": {"library": "WikiPathways_2024_Mouse", "version": "2024 (most recent Enrichr release; no 2026 release exists)", "available": True},
            "MSigDB (Hallmark)": {"library": None, "version": None, "available": False,
                       "note": "Enrichr's MSigDB Hallmark library is human-only. Use STRING Enrichment Analysis for mouse."},
        },
    },
    "Rat (Rattus norvegicus)": {
        "string_taxid": 10116,
        "enrichr_base": None,
        "gene_id_type": "Gene Symbol (RGD)",
        "databases": {c: {"library": None, "version": None, "available": False,
                           "note": "Enrichr has no rat-specific gene set libraries or organism portal (there is no 'RatEnrichr'). Use STRING Enrichment Analysis, which supports rat directly."}
                      for c in ["GO Biological Process", "GO Molecular Function", "GO Cellular Component",
                                "KEGG", "Reactome", "WikiPathways", "MSigDB (Hallmark)"]},
    },
    "Yeast (S. cerevisiae)": {
        "string_taxid": 4932,
        "enrichr_base": YEAST_ENRICHR_BASE,
        "gene_id_type": "Gene Symbol (SGD)",
        "databases": {
            "GO Biological Process": {"library_prefix": "GO_Biological_Process", "version": "current YeastEnrichr release (organism-specific Enrichr instance; not year-stamped -- exact name resolved and logged at query time)", "available": True},
            "GO Molecular Function": {"library_prefix": "GO_Molecular_Function", "version": "current YeastEnrichr release (not year-stamped)", "available": True},
            "GO Cellular Component": {"library_prefix": "GO_Cellular_Component", "version": "current YeastEnrichr release (not year-stamped)", "available": True},
            "KEGG": {"library": None, "version": None, "available": False,
                     "note": "No confirmed KEGG library on YeastEnrichr. Use STRING Enrichment Analysis for yeast KEGG-style annotation."},
            "Reactome": {"library": None, "version": None, "available": False,
                         "note": "No confirmed Reactome library on YeastEnrichr."},
            "WikiPathways": {"library_prefix": "WikiPathways", "version": "current YeastEnrichr release (not year-stamped)", "available": True},
            "MSigDB (Hallmark)": {"library": None, "version": None, "available": False, "note": "No MSigDB library on YeastEnrichr."},
        },
    },
    "Zebrafish (Danio rerio)": {
        "string_taxid": 7955,
        "enrichr_base": FISH_ENRICHR_BASE,
        "gene_id_type": "Gene Symbol (ZFIN)",
        "databases": {
            "GO Biological Process": {"library_prefix": "GO_Biological_Process", "version": "current FishEnrichr release (organism-specific Enrichr instance; not year-stamped -- exact name resolved and logged at query time)", "available": True},
            "GO Molecular Function": {"library_prefix": "GO_Molecular_Function", "version": "current FishEnrichr release (not year-stamped)", "available": True},
            "GO Cellular Component": {"library_prefix": "GO_Cellular_Component", "version": "current FishEnrichr release (not year-stamped)", "available": True},
            "KEGG": {"library": None, "version": None, "available": False,
                     "note": "No confirmed KEGG library on FishEnrichr. Use STRING Enrichment Analysis for zebrafish KEGG-style annotation."},
            "Reactome": {"library": None, "version": None, "available": False, "note": "No confirmed Reactome library on FishEnrichr."},
            "WikiPathways": {"library_prefix": "WikiPathways", "version": "current FishEnrichr release (not year-stamped)", "available": True},
            "MSigDB (Hallmark)": {"library": None, "version": None, "available": False, "note": "No MSigDB library on FishEnrichr."},
        },
    },
}

# Major plant species: no Enrichr-family portal exists for any plant genome, so
# ORA/GSEA (Enrichr-backed) are unavailable for all of them -- STRING Enrichment
# Analysis is the supported path, since STRING itself covers these organisms.
_PLANT_SPECIES = {
    "Arabidopsis thaliana": 3702, "Rice (Oryza sativa)": 39947,
    "Maize (Zea mays)": 4577, "Soybean (Glycine max)": 3847,
    "Tomato (Solanum lycopersicum)": 4081,
}
for _name, _taxid in _PLANT_SPECIES.items():
    ORGANISM_CATALOG[f"{_name} (plant)"] = {
        "string_taxid": _taxid, "enrichr_base": None, "gene_id_type": "Gene Symbol / Locus ID",
        "databases": {c: {"library": None, "version": None, "available": False,
                           "note": "No Enrichr-family portal exists for plant species. Use STRING Enrichment Analysis, which supports this organism directly via STRING's own database."}
                      for c in ["GO Biological Process", "GO Molecular Function", "GO Cellular Component",
                                "KEGG", "Reactome", "WikiPathways", "MSigDB (Hallmark)"]},
    }

# Kept for the STRING Enrichment Analysis tab's species selector (STRING supports
# every organism above, and many more, directly -- this dict is intentionally the
# same set surfaced for both STRING calls and the ORA/GSEA organism selector).
STRING_SPECIES = {name: cfg["string_taxid"] for name, cfg in ORGANISM_CATALOG.items()}

ORA_GSEA_DATABASE_COLLECTIONS = ["GO Biological Process", "GO Molecular Function", "GO Cellular Component",
                                  "KEGG", "Reactome", "WikiPathways", "MSigDB (Hallmark)"]


def get_database_info(organism: str, collection: str) -> dict:
    """Look up one organism/collection cell of ORGANISM_CATALOG."""
    org = ORGANISM_CATALOG.get(organism)
    if org is None:
        raise EnrichmentError(f"Unsupported organism: '{organism}'.")
    info = org["databases"].get(collection)
    if info is None:
        raise EnrichmentError(f"Unsupported database collection: '{collection}'.")
    return info


# ---------------------------------------------------------------------------
# Method 1: STRING Enrichment Analysis
# ---------------------------------------------------------------------------
def get_string_version():
    """Current STRING database version + stable API address (live, per-request)."""
    try:
        resp = requests.get(f"{STRING_BASE}/tsv/version", timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        if len(df):
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return None


# The exact, fixed column set (and order) STRING enrichment results are reported
# in throughout the app -- selected and reordered from STRING's raw TSV response,
# which also includes fields (category, term, preferredNames) not part of this
# output.
STRING_OUTPUT_COLUMNS = ["description", "p_value", "fdr", "number_of_genes",
                          "number_of_genes_in_background", "ncbiTaxonId", "inputGenes"]


def run_string_enrichment(gene_list, species=9606):
    """
    Functional enrichment via the STRING database API. STRING computes this
    server-side, hypergeometric-style, against its own bundled annotation sets
    -- see get_string_version() for the exact database vintage behind these
    results. STRING's raw response includes additional fields (category, term,
    preferredNames); this returns only, and always in this exact order:
        description, p_value, fdr, number_of_genes,
        number_of_genes_in_background, ncbiTaxonId, inputGenes
    If STRING's response is ever missing one of these (e.g. an API change),
    that column comes back filled with NaN rather than silently disappearing,
    so callers can rely on this schema being stable.
    """
    genes = [str(g).strip() for g in gene_list if str(g).strip()]
    if len(genes) < 1:
        raise EnrichmentError("Need at least 1 gene for STRING enrichment.")
    try:
        resp = requests.post(
            f"{STRING_BASE}/tsv/enrichment",
            data={"identifiers": "%0d".join(genes), "species": species,
                  "caller_identity": "ProteoAI_Pro"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EnrichmentError(
            f"Could not reach STRING (string-db.org) — check your internet connection. "
            f"Details: {e}"
        )
    text = resp.text.strip()
    if not text or text.startswith("Error"):
        return pd.DataFrame(columns=STRING_OUTPUT_COLUMNS)
    raw = pd.read_csv(io.StringIO(text), sep="\t")
    return raw.reindex(columns=STRING_OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Gene set library retrieval (shared by ORA and GSEA)
# ---------------------------------------------------------------------------
def get_available_libraries(enrichr_base: str = ENRICHR_BASE):
    """
    Live list of every gene set library available on a given Enrichr-family
    instance (main Enrichr, YeastEnrichr, FishEnrichr, ...), via that instance's
    `datasetStatistics` endpoint. Used to resolve organism-specific instances'
    un-dated library names (see resolve_library_name()) and to fail loudly
    rather than guess when a requested library doesn't actually exist there.
    """
    try:
        resp = requests.get(f"{enrichr_base}/datasetStatistics", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        raise EnrichmentError(f"Could not reach {enrichr_base} — check your internet connection. Details: {e}")
    except ValueError as e:
        raise EnrichmentError(f"Unexpected response listing libraries from {enrichr_base}: {e}")
    return [entry["libraryName"] for entry in data.get("statistics", [])]


def resolve_library_name(enrichr_base: str, prefix: str) -> str:
    """
    Find the best-matching real library name on `enrichr_base` starting with
    `prefix` (case-insensitive), preferring the lexicographically LAST match
    (year-stamped names sort so the most recent year wins; un-dated names are
    returned as-is when only one match exists). Raises EnrichmentError -- never
    silently falls back to a fabricated name -- if nothing matches.
    """
    available = get_available_libraries(enrichr_base)
    matches = sorted(name for name in available if name.lower().startswith(prefix.lower()))
    if not matches:
        raise EnrichmentError(
            f"No library starting with '{prefix}' was found on {enrichr_base}. "
            f"This resource may not provide that collection for this organism."
        )
    return matches[-1]


def fetch_gene_set_library(organism: str, collection: str):
    """
    Resolve and fetch the GMT gene set library for one (organism, collection)
    cell of ORGANISM_CATALOG, for use by both ORA (Method 2) and GSEA (Method 3).

    Returns (gene_sets: {term: [gene_symbols]}, resolved_library_name: str,
    documented_version: str). Raises EnrichmentError with the catalog's
    documented reason if the combination is unsupported -- never substitutes a
    different organism's/collection's library silently.
    """
    info = get_database_info(organism, collection)
    if not info.get("available", False):
        raise EnrichmentError(
            f"'{collection}' is not available for {organism}: "
            f"{info.get('note', 'no database available for this combination.')}"
        )
    org_cfg = ORGANISM_CATALOG[organism]
    base = org_cfg["enrichr_base"]
    if base is None:
        raise EnrichmentError(f"No Enrichr-family instance configured for {organism}.")

    if info.get("library"):
        library_name = info["library"]
    else:
        library_name = resolve_library_name(base, info["library_prefix"])

    gene_sets = get_enrichr_gene_set_library(library_name, enrichr_base=base)
    return gene_sets, library_name, info["version"]


def get_enrichr_gene_set_library(library: str, enrichr_base: str = ENRICHR_BASE):
    """
    Fetch a full gene set library in GMT format from any Enrichr-family
    instance (main Enrichr by default; pass enrichr_base for YeastEnrichr/
    FishEnrichr) and parse it into {term: [gene_symbols]}.
    """
    try:
        resp = requests.get(
            f"{enrichr_base}/geneSetLibrary",
            params={"mode": "text", "libraryName": library}, timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EnrichmentError(
            f"Could not reach {enrichr_base} — check your internet connection. Details: {e}"
        )
    if not resp.text.strip():
        raise EnrichmentError(f"Library '{library}' returned no data from {enrichr_base}.")
    return parse_gmt_text(resp.text)


def parse_gmt_text(text: str) -> dict:
    """
    Parse GMT-format text (the standard MSigDB/Enrichr convention:
    term<TAB>description<TAB>gene1<TAB>gene2<TAB>...) into {term: [gene_symbols]}.
    """
    gene_sets = {}
    for line in text.strip().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        term = parts[0].strip()
        genes = [g.strip().upper() for g in parts[2:] if g.strip()]
        if genes:
            gene_sets[term] = genes
    return gene_sets


# ---------------------------------------------------------------------------
# Method 2: Over-Representation Analysis (ORA) -- hypergeometric test, computed
# locally with an explicit, reported universe. This is a classic contingency-
# table enrichment test and is NOT the same algorithm as GSEA (Method 3): ORA
# takes one fixed comparison list and asks "is this gene set over-represented in
# it?"; GSEA (below) instead evaluates every gene's position in a full ranking.
# The two must never be conflated or substituted for one another.
# ---------------------------------------------------------------------------
ORA_RESULT_COLUMNS = ["Gene Set", "P-value", "FDR", "# genes in universe", "# Genes in Gene Set",
                      "# genes in comparison", "# Genes in Overlap", "Fold Enrichment", "Overlapped genes"]


def run_ora_hypergeometric(comparison_genes, gene_sets: dict, universe_genes=None,
                            min_set_size: int = 1, max_set_size: int = None):
    """
    Over-Representation Analysis via the hypergeometric test.

    comparison_genes: the FULL user-provided comparison/input gene list -- used
        exactly as given, with no additional significance/rank cutoff applied
        inside this function. (Any cutoff used to arrive at this list, e.g. an
        FDR threshold in the Statistics tab, is the caller's own upstream choice,
        made once, before calling this function -- not something this function
        does on its own.)
    gene_sets: {term: [gene_symbols]}, e.g. from fetch_gene_set_library().
    universe_genes: the background/universe gene list. If None, defaults to the
        union of every gene appearing anywhere in `gene_sets` (documented as
        such in the returned metadata). Passing the set of proteins actually
        DETECTED in the experiment -- the recommended default wired in the UI --
        is generally the more defensible universe, since it reflects genes that
        could actually have been observed, not the whole genome.
    min_set_size / max_set_size: skip gene sets whose overlap with the universe
        falls outside this range (very small or very large gene sets can be
        excluded from the report; defaults keep everything with >=1 gene).

    For each gene set that clears the size filter:
        N = |universe|
        K = |gene set ∩ universe|
        n = |comparison ∩ universe|
        k = |comparison ∩ gene set ∩ universe|
        P-value = P(X >= k) = hypergeom.sf(k-1, N, K, n)   (one-sided upper tail:
            probability of seeing this many or more overlapping genes by chance)
        Fold Enrichment = (k/n) / (K/N)
    FDR = Benjamini-Hochberg, applied across every gene set actually tested here
    (not across the full library, and not silently limited to "significant" ones).

    Returns (result_df, run_meta):
        result_df has EXACTLY these columns, one row per tested gene set:
        Gene Set, P-value, FDR, # genes in universe, # Genes in Gene Set,
        # genes in comparison, # Genes in Overlap, Fold Enrichment, Overlapped genes
        (Overlapped genes are ';'-delimited.)
        run_meta is a dict of reproducibility metadata (background source and
        size, gene counts before/after universe mapping, statistical test name,
        correction method).
    """
    if not gene_sets:
        raise EnrichmentError("No gene sets to test (the selected library returned nothing).")

    comparison_raw = [str(g).strip() for g in comparison_genes if str(g).strip()]
    if len(comparison_raw) == 0:
        raise EnrichmentError("Comparison gene list is empty.")
    comparison = set(g.upper() for g in comparison_raw)
    n_duplicates = len(comparison_raw) - len(comparison)

    all_library_genes = set()
    for genes in gene_sets.values():
        all_library_genes.update(g.upper() for g in genes)

    if universe_genes is not None:
        universe_raw = [str(g).strip() for g in universe_genes if str(g).strip()]
        universe = set(g.upper() for g in universe_raw)
        universe_source = f"user-provided background ({len(universe)} genes -- e.g. all proteins detected in this study)"
    else:
        universe = set(all_library_genes)
        universe_source = f"union of all genes in the selected gene-set library ({len(universe)} genes)"

    comparison_in_universe = comparison & universe
    n_comparison_dropped = len(comparison) - len(comparison_in_universe)

    N = len(universe)
    n = len(comparison_in_universe)

    rows = []
    n_sets_skipped_size = 0
    if n > 0:
        for term, genes in gene_sets.items():
            gene_set_genes = set(g.upper() for g in genes) & universe
            K = len(gene_set_genes)
            if K == 0:
                continue
            if K < min_set_size or (max_set_size is not None and K > max_set_size):
                n_sets_skipped_size += 1
                continue
            overlap = comparison_in_universe & gene_set_genes
            k = len(overlap)
            pval = hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0
            fold_enrichment = (k / n) / (K / N) if (n > 0 and K > 0 and N > 0) else 0.0
            rows.append({
                "Gene Set": term, "P-value": pval, "# genes in universe": N,
                "# Genes in Gene Set": K, "# genes in comparison": n,
                "# Genes in Overlap": k, "Fold Enrichment": fold_enrichment,
                "Overlapped genes": ";".join(sorted(overlap)),
            })

    result_df = pd.DataFrame(rows, columns=[c for c in ORA_RESULT_COLUMNS if c != "FDR"])
    if len(result_df):
        result_df["FDR"] = multipletests(result_df["P-value"], method="fdr_bh")[1]
        result_df = result_df[ORA_RESULT_COLUMNS].sort_values("P-value").reset_index(drop=True)
    else:
        result_df = pd.DataFrame(columns=ORA_RESULT_COLUMNS)

    run_meta = {
        "Analysis method": "Over-Representation Analysis (ORA)",
        "Statistical test": "Hypergeometric test, one-sided (P(X >= k overlapping genes))",
        "Multiple-testing correction": "Benjamini-Hochberg FDR",
        "Background/universe definition": universe_source,
        "# genes in universe (N)": N,
        "# genes in input comparison list": len(comparison_raw),
        "# duplicate genes in input (collapsed)": n_duplicates,
        "# genes in comparison list mapped to universe (n)": n,
        "# genes in comparison list NOT found in universe (discarded)": n_comparison_dropped,
        "# gene sets in library": len(gene_sets),
        "# gene sets tested (after size filter)": len(result_df),
        "# gene sets skipped (outside min/max size)": n_sets_skipped_size,
    }
    return result_df, run_meta


# ---------------------------------------------------------------------------
# Method 3: Gene Set Enrichment Analysis (GSEA) -- classic weighted running-sum
# statistic (Subramanian et al. 2005). Statistically distinct from ORA (Method
# 2, above): GSEA never applies a significance cutoff to build its input --
# every detected protein is ranked and used.
# ---------------------------------------------------------------------------
def compute_ranking_score(stats_df: pd.DataFrame, method: str = "signed_neglogp") -> pd.Series:
    """
    Build the per-protein ranking score for GSEA from a Statistics-tab-style
    table (must contain a Log2FC-like column and a PValue-like column). Every
    row of `stats_df` is used -- GSEA ranks the complete detected protein list,
    not a pre-filtered significant subset.

    method:
      'signed_neglogp' : sign(log2FC) * -log10(p-value) -- the standard choice,
                          combining direction and significance (as used
                          throughout the GSEAPY prerank documentation).
      'log2fc'         : log2 fold change alone.
      'statistic'      : the test statistic column, if present (e.g. t-statistic),
                          used as-is (already signed).
    """
    lfc_col = next((c for c in stats_df.columns if c.lower() in ("log2fc", "log2_fold_change")), None)
    p_col = next((c for c in stats_df.columns if c.lower() in ("pvalue", "p_value", "p-value")), None)
    stat_col = next((c for c in stats_df.columns if c.lower() in ("statistic", "t-statistic", "t_statistic", "score")), None)

    if method == "statistic" and stat_col is not None:
        score = stats_df[stat_col].astype(float)
    elif method == "log2fc":
        if lfc_col is None:
            raise EnrichmentError("Statistics table has no Log2FC column.")
        score = stats_df[lfc_col].astype(float)
    else:
        if lfc_col is None or p_col is None:
            raise EnrichmentError("Statistics table must contain a Log2FC and a PValue column.")
        lfc = stats_df[lfc_col].astype(float)
        pval = stats_df[p_col].astype(float).clip(lower=1e-300)
        score = np.sign(lfc) * -np.log10(pval)

    score = score.replace([np.inf, -np.inf], 0).fillna(0)
    score.index = score.index.map(lambda g: str(g).upper())
    return score


def _weighted_running_sum(ranked_scores: np.ndarray, hit_mask: np.ndarray, weight: float = 1.0):
    """
    Core Subramanian et al. 2005 weighted Kolmogorov-Smirnov running-sum statistic.
    Returns (ES, running_sum_array).
    """
    n = len(ranked_scores)
    n_hits = int(hit_mask.sum())
    if n_hits == 0:
        return 0.0, np.zeros(n)
    abs_scores = np.abs(ranked_scores) ** weight
    hit_sum = abs_scores[hit_mask].sum()
    if hit_sum == 0:
        hit_step = np.where(hit_mask, 1.0 / n_hits, 0.0)
    else:
        hit_step = np.where(hit_mask, abs_scores / hit_sum, 0.0)
    n_miss = n - n_hits
    miss_step = np.where(~hit_mask, 1.0 / max(n_miss, 1), 0.0)
    running_sum = np.cumsum(hit_step - miss_step)
    peak_idx = int(np.argmax(np.abs(running_sum)))
    es = running_sum[peak_idx]
    return es, running_sum


def run_prerank_gsea(ranked_scores: pd.Series, gene_sets: dict, weight: float = 1.0,
                      n_perm: int = 1000, min_size: int = 15, max_size: int = 500,
                      seed: int = 42):
    """
    Gene Set Enrichment Analysis (Subramanian et al. 2005), preranked mode.
    ranked_scores: Series indexed by gene symbol, EVERY detected protein (from
    compute_ranking_score() -- no significance cutoff applied). gene_sets:
    {term: [genes]}.

    Handles duplicate gene symbols in the ranking by keeping the entry with the
    largest-magnitude score per symbol (the most informative one, if the same
    symbol appears more than once after ID mapping) and drops rows with missing
    (NaN) scores before ranking, both reported in run_meta.

    Significance is estimated via GENE-SET permutation (shuffle which ranked
    positions count as "hits", same set size, recompute ES, repeat n_perm times)
    -- the same approach GSEAPY's prerank mode uses, since the original
    Subramanian et al. phenotype-permutation procedure needs the per-sample
    expression matrix and group labels, which aren't available from a ranked
    list alone. This is a genuinely different statistical test from the
    hypergeometric test used by ORA (Method 2) -- not a relabeling of it.

    Returns (result_df, run_meta). result_df: Term, Size, ES, NES, P-value, FDR
    (BH across tested gene sets), Leading_Edge_Size, Leading_Edge (leading-edge
    genes, ';'-delimited, capped at 50 for display).
    """
    if not gene_sets:
        raise EnrichmentError("No gene sets to test (the selected library returned nothing).")

    n_before = len(ranked_scores)
    ranked_scores = ranked_scores.dropna()
    n_missing = n_before - len(ranked_scores)
    if ranked_scores.index.duplicated().any():
        # Keep the largest-magnitude score per duplicated gene symbol (the most
        # informative entry, if the same symbol appears more than once after ID
        # mapping -- e.g. two protein isoforms mapping to the same gene).
        tmp = ranked_scores.to_frame(name="score")
        tmp["abs_score"] = tmp["score"].abs()
        tmp = tmp.sort_values("abs_score", ascending=False)
        tmp = tmp[~tmp.index.duplicated(keep="first")]
        ranked_scores = tmp["score"]
    n_duplicates_collapsed = n_before - n_missing - len(ranked_scores)

    if len(ranked_scores) < min_size:
        raise EnrichmentError(
            f"Only {len(ranked_scores)} ranked genes available (after dropping missing/duplicate "
            f"entries) -- need at least {min_size} (the minimum gene set size) for GSEA to run."
        )

    ranked_scores = ranked_scores.sort_values(ascending=False)
    genes_ranked = ranked_scores.index.to_numpy()
    scores_arr = ranked_scores.to_numpy(dtype=float)
    n = len(genes_ranked)
    gene_pos = {g: i for i, g in enumerate(genes_ranked)}

    rng = np.random.default_rng(seed)
    results = []
    n_sets_too_small = 0
    n_sets_too_large = 0
    for term, term_genes in gene_sets.items():
        hit_idx = [gene_pos[g] for g in term_genes if g in gene_pos]
        size = len(hit_idx)
        if size < min_size:
            n_sets_too_small += 1
            continue
        if size > max_size:
            n_sets_too_large += 1
            continue
        hit_mask = np.zeros(n, dtype=bool)
        hit_mask[hit_idx] = True
        es, running_sum = _weighted_running_sum(scores_arr, hit_mask, weight)

        null_es = np.empty(n_perm)
        for p in range(n_perm):
            perm_idx = rng.choice(n, size=size, replace=False)
            perm_mask = np.zeros(n, dtype=bool)
            perm_mask[perm_idx] = True
            null_es[p], _ = _weighted_running_sum(scores_arr, perm_mask, weight)

        if es >= 0:
            pos_null = null_es[null_es >= 0]
            denom = pos_null.mean() if pos_null.size and pos_null.mean() != 0 else 1.0
            nes = es / denom
            pval = ((pos_null >= es).sum() + 1) / (pos_null.size + 1) if pos_null.size else 1.0
        else:
            neg_null = null_es[null_es < 0]
            denom = abs(neg_null.mean()) if neg_null.size and neg_null.mean() != 0 else 1.0
            nes = es / denom
            pval = ((neg_null <= es).sum() + 1) / (neg_null.size + 1) if neg_null.size else 1.0

        peak_idx = int(np.argmax(np.abs(running_sum)))
        if es >= 0:
            le_genes = [genes_ranked[i] for i in hit_idx if i <= peak_idx]
        else:
            le_genes = [genes_ranked[i] for i in hit_idx if i >= peak_idx]

        results.append({
            "Term": term, "Size": size, "ES": es, "NES": nes, "P-value": pval,
            "Leading_Edge_Size": len(le_genes), "Leading_Edge": ";".join(le_genes[:50]),
        })

    if not results:
        result_df = pd.DataFrame(columns=["Term", "Size", "ES", "NES", "P-value", "FDR",
                                           "Leading_Edge_Size", "Leading_Edge"])
    else:
        result_df = pd.DataFrame(results)
        result_df["FDR"] = multipletests(result_df["P-value"], method="fdr_bh")[1]
        result_df = result_df.sort_values("NES", ascending=False).reset_index(drop=True)

    run_meta = {
        "Analysis method": "Gene Set Enrichment Analysis (GSEA), preranked (Subramanian et al. 2005)",
        "Statistical test": "Weighted Kolmogorov-Smirnov running-sum statistic (ES), normalized to NES",
        "Significance estimation": f"Gene-set permutation, {n_perm} permutations per gene set",
        "Multiple-testing correction": "Benjamini-Hochberg FDR",
        "# genes in ranked list (input)": n_before,
        "# genes dropped (missing ranking value)": n_missing,
        "# duplicate gene symbols collapsed (kept largest-magnitude score)": n_duplicates_collapsed,
        "# genes in ranked list (used)": n,
        "# gene sets in library": len(gene_sets),
        "# gene sets tested": len(result_df),
        "# gene sets skipped (below min size)": n_sets_too_small,
        "# gene sets skipped (above max size)": n_sets_too_large,
        "Min/Max gene set size": f"{min_size} / {max_size}",
    }
    return result_df, run_meta


def running_score_plot(ranked_scores: pd.Series, gene_sets: dict, term: str, weight: float = 1.0,
                        nes: float = None, pvalue: float = None, fdr: float = None):
    """
    The classic GSEA 'mountain plot' for one gene set: running enrichment score
    across the ranked list (top), a barcode of hit positions (middle), and the
    ranking metric itself (bottom) -- the same 3-panel layout used by the Broad
    Institute's GSEA desktop tool and reproduced by GSEAPY's plotting functions.

    nes/pvalue/fdr (optional): this term's NES, P-value, and FDR from the
    corresponding row of run_prerank_gsea()'s result_df -- when supplied, they're
    displayed as a header line above the plot (as NES/pVal/FDR) so the term's
    full significance profile is visible without cross-referencing the results
    table. ES itself is not shown (NES is the standardized, comparable-across-
    gene-set-sizes version and is what a reader should actually judge by).
    """
    if term not in gene_sets:
        raise EnrichmentError(f"Gene set '{term}' not found.")
    ranked_scores = ranked_scores.dropna().sort_values(ascending=False)
    genes_ranked = ranked_scores.index.to_numpy()
    scores_arr = ranked_scores.to_numpy(dtype=float)
    n = len(genes_ranked)
    gene_pos = {g: i for i, g in enumerate(genes_ranked)}
    hit_idx = sorted(gene_pos[g] for g in gene_sets[term] if g in gene_pos)
    if not hit_idx:
        raise EnrichmentError(f"None of '{term}'s genes were found in the ranked list.")
    hit_mask = np.zeros(n, dtype=bool)
    hit_mask[hit_idx] = True
    es, running_sum = _weighted_running_sum(scores_arr, hit_mask, weight)
    peak_idx = int(np.argmax(np.abs(running_sum)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True,
                              gridspec_kw={"height_ratios": [3, 0.6, 1.5], "hspace": 0.08})
    axes[0].plot(range(n), running_sum, color="#2E7D32", linewidth=1.6)
    axes[0].axhline(0, color="gray", linewidth=0.6)
    axes[0].axvline(peak_idx, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Running Enrichment Score")

    stat_parts = []
    if nes is not None:
        stat_parts.append(f"NES = {nes:.3f}")
    if pvalue is not None:
        stat_parts.append(f"pVal = {pvalue:.3g}")
    if fdr is not None:
        stat_parts.append(f"FDR = {fdr:.3g}")
    fig.suptitle(term, fontsize=10.5, fontweight="bold", y=0.985)
    if stat_parts:
        axes[0].set_title("   |   ".join(stat_parts), fontsize=9.5, color="#333333")

    axes[1].vlines(hit_idx, 0, 1, color="black", linewidth=0.6)
    axes[1].set_yticks([])
    axes[1].set_ylabel("Hits", fontsize=8)

    axes[2].fill_between(range(n), scores_arr, color="#4C72B0", alpha=0.7, linewidth=0)
    axes[2].axhline(0, color="gray", linewidth=0.6)
    axes[2].set_ylabel("Ranking Score")
    axes[2].set_xlabel("Rank in ordered protein list")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig



def _enrichment_dotplot(result_df: pd.DataFrame, term_col: str, x_col: str, size_col: str,
                         color_col: str, x_label: str, title: str, size_legend_label: str = "Count",
                         color_legend_label: str = None, top_n: int = 15, reference_x: float = None,
                         select_col: str = None, select_ascending: bool = True):
    """
    Shared rendering engine behind gsea_dotplot(), ora_dotplot(), and
    string_dotplot() -- one consistent visual style across all three enrichment
    methods: a single panel, pathways ordered along the y-axis by x_col (so the
    strongest hit sits at the top), dot size encoding a gene-count column, and
    dot color encoding color_col via a blue (low) -> red (high) colormap, with
    a color bar and a size legend to the right of the plot.

    Selects the top_n rows by select_col (defaults to color_col if not given),
    then orders that selection along the y-axis by x_col ascending. select_col
    is kept separate from color_col so which rows are "most significant" (by
    raw FDR, smaller = better) can be decided independently of what's actually
    drawn as the dot color (e.g. a -log10(FDR) transform of that same FDR
    column, where larger = better) -- selecting by the transformed column with
    the same ascending=True would keep the LEAST significant rows instead.

    term_col/x_col/size_col/color_col: column names in result_df to use for the
    pathway label, x-axis value, dot size, and dot color respectively.
    reference_x: if given, draws a dashed vertical reference line at this x
    value (0 for GSEA's NES, 1.0 for ORA/STRING's fold-enrichment-style ratios
    where "no enrichment" has a natural non-zero value; omit for ratios with no
    single natural reference point).
    """
    select_col = select_col or color_col
    if result_df is None or len(result_df) == 0:
        raise EnrichmentError("No enrichment results to plot.")
    required = {term_col, x_col, size_col, color_col, select_col}
    missing = required - set(result_df.columns)
    if missing:
        raise EnrichmentError(f"Enrichment results are missing required column(s): {', '.join(sorted(missing))}")

    work = result_df.dropna(subset=[x_col, size_col, color_col, select_col]).copy()
    if len(work) == 0:
        raise EnrichmentError("No enrichment results with complete (non-missing) data to plot.")

    subset = work.sort_values(select_col, ascending=select_ascending).head(top_n)
    subset = subset.sort_values(x_col, ascending=True)
    terms = subset[term_col].astype(str).tolist()
    n = len(terms)
    if n == 0:
        raise EnrichmentError("No gene sets available to plot.")

    fig, ax = plt.subplots(figsize=(9.5, max(4.6, 0.5 * n + 1.3)))
    cmap = matplotlib.colormaps.get_cmap("coolwarm")  # blue = low value, red = high value
    vmin, vmax = float(subset[color_col].min()), float(subset[color_col].max())
    if vmin == vmax:
        vmax = vmin + 1e-9
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    size_min, size_max = float(subset[size_col].min()), float(subset[size_col].max())

    def _scale(v):
        if size_max == size_min:
            return 220.0
        return 80.0 + 420.0 * (v - size_min) / (size_max - size_min)

    y_pos = np.arange(n)

    # Subtle alternating row bands behind the dots -- a standard readability aid
    # in professional dot plots (clusterProfiler, ggplot2 themes) that makes it
    # much easier to trace a given pathway's row across to its y-axis label,
    # especially with many rows.
    for i in range(n):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color="#F4F4F4", zorder=0)

    ax.scatter(subset[x_col], y_pos, s=subset[size_col].apply(_scale),
               c=subset[color_col], cmap=cmap, norm=norm,
               edgecolor="black", linewidth=0.5, zorder=3)

    if reference_x is not None:
        ax.axvline(reference_x, color="#444444", linestyle="--", linewidth=1.0, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=10)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(x_label, fontsize=11.5, fontweight="bold")
    ax.set_title(title, fontsize=15.5, fontweight="bold", pad=14)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Left margin sized to comfortably fit the longest pathway name -- a fixed
    # margin either clips long names or wastes space with short ones.
    max_label_len = max((len(t) for t in terms), default=10)
    left_margin = min(0.42, max(0.16, 0.011 * max_label_len))
    fig.subplots_adjust(left=left_margin, right=0.76)

    # Legend panel: a dedicated size-legend axes sitting directly above the
    # colorbar with almost no gap between them, both anchored together low in
    # the bottom-right corner of the figure -- one compact, borderless legend
    # block (no bounding box) with the size dots packed close together, matching
    # the clean look of built-in ggplot2/R-style legends rather than a boxed,
    # loosely-spaced matplotlib default.
    #
    # The size legend is drawn manually on its own small axes (not via
    # matplotlib's automatic Legend) because Legend's row spacing is based on
    # font size, not marker size: when the size range being illustrated is
    # wide, the largest marker overflows into neighboring rows and the three
    # dots visually merge into a "snowman". Drawing them explicitly at fixed,
    # tightly-but-safely-separated y-positions guarantees they stay distinct
    # dots no matter how large the biggest one is. Each legend dot is plotted
    # with the SAME `s=` (points^2) value the real data dots use, so the
    # legend is a true, honest representation of size -- not a separately-
    # scaled stand-in.
    reps = sorted(set(int(round(v)) for v in np.linspace(size_min, size_max, num=3)))
    legend_ax = fig.add_axes([0.80, 0.275, 0.16, 0.19])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_frame_on(False)  # borderless -- no bounding box around the dots
    legend_ax.set_title(size_legend_label, fontsize=9.5, fontweight="bold", loc="left", pad=4)

    dot_x = 0.30
    y_positions = np.linspace(0.82, 0.32, len(reps)) if len(reps) > 1 else [0.5]
    for y, c in zip(y_positions, reps):
        legend_ax.scatter([dot_x], [y], s=_scale(c), c="#555555",
                          edgecolor="black", linewidth=0.5, clip_on=False, zorder=3)
        legend_ax.text(dot_x + 0.30, y, str(c), fontsize=9, va="center", ha="left")

    # Color bar, titled above it, placed almost flush against the size legend
    # panel above (a hairline gap only) so the two read as one legend block.
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.835, 0.05, 0.03, 0.17])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title(color_legend_label or color_col, fontsize=10.5, fontweight="bold", pad=8)

    return fig


def _neglog10_fdr(fdr_series: pd.Series) -> pd.Series:
    """
    -log10(FDR), the standard significance transform for enrichment dot-plot
    color scales (larger = more significant, unlike raw FDR where smaller =
    more significant). FDR values of exactly 0 are floored to the smallest
    representable positive float first, so -log10 never produces +inf.
    """
    return -np.log10(fdr_series.astype(float).clip(lower=np.finfo(float).tiny))


def gsea_dotplot(result_df: pd.DataFrame, top_n: int = 15):
    """
    GSEA pathway dot plot: x-axis is NES (Normalized Enrichment Score, positive
    = activated, negative = suppressed, dashed reference line at 0), dot size
    is Leading Edge Size, dot color is -log10(FDR) (computed here from the
    FDR column; larger = more significant). Shows the top_n gene sets by raw
    FDR (most significant first), ordered top-to-bottom from highest to lowest NES.
    """
    if result_df is None or len(result_df) == 0:
        raise EnrichmentError("No GSEA results to plot.")
    if "FDR" not in result_df.columns:
        raise EnrichmentError("GSEA results are missing the FDR column needed for the color scale.")
    df = result_df.copy()
    df["-log10(FDR)"] = _neglog10_fdr(df["FDR"])
    return _enrichment_dotplot(
        df, term_col="Term", x_col="NES", size_col="Leading_Edge_Size", color_col="-log10(FDR)",
        x_label="Normalized Enrichment Score (NES)", title="GSEA Enrichment Dot Plot",
        size_legend_label="Leading Edge Size", color_legend_label="-log10(FDR)",
        top_n=top_n, reference_x=0.0, select_col="FDR", select_ascending=True,
    )


def ora_dotplot(result_df: pd.DataFrame, top_n: int = 15):
    """
    ORA pathway dot plot, in the same visual style as gsea_dotplot(): x-axis is
    Fold Enrichment (dashed reference line at 1.0, the "exactly as expected by
    chance" value), dot size is Genes in Overlap, dot color is -log10(FDR)
    (computed here from the FDR column; larger = more significant). ORA has no
    direction/sign the way GSEA's NES does, so gene sets are simply ordered by
    Fold Enrichment; the top_n shown are the most significant by raw FDR.
    """
    if result_df is None or len(result_df) == 0:
        raise EnrichmentError("No ORA results to plot.")
    if "FDR" not in result_df.columns:
        raise EnrichmentError("ORA results are missing the FDR column needed for the color scale.")
    df = result_df.copy()
    df["-log10(FDR)"] = _neglog10_fdr(df["FDR"])
    return _enrichment_dotplot(
        df, term_col="Gene Set", x_col="Fold Enrichment", size_col="# Genes in Overlap",
        color_col="-log10(FDR)", x_label="Fold Enrichment", title="ORA Enrichment Dot Plot",
        size_legend_label="Genes in Overlap", color_legend_label="-log10(FDR)",
        top_n=top_n, reference_x=1.0, select_col="FDR", select_ascending=True,
    )


def string_dotplot(result_df: pd.DataFrame, top_n: int = 15):
    """
    STRING pathway dot plot, in the same visual style as gsea_dotplot(): x-axis
    is Gene Ratio (number_of_genes / number_of_genes_in_background, computed
    here), dot size is Gene Count, dot color is -log10(FDR) (computed here from
    STRING's fdr column; larger = more significant). No single reference
    x-value applies to a 0-1 ratio with no companion "expected under chance"
    figure, so no dashed reference line is drawn.
    """
    if result_df is None or len(result_df) == 0:
        raise EnrichmentError("No STRING enrichment results to plot.")
    if "number_of_genes_in_background" not in result_df.columns or "number_of_genes" not in result_df.columns:
        raise EnrichmentError(
            "STRING results are missing the gene-count columns needed to compute Gene Ratio."
        )
    if "fdr" not in result_df.columns:
        raise EnrichmentError("STRING results are missing the fdr column needed for the color scale.")
    df = result_df.copy()
    df["Gene Ratio"] = df["number_of_genes"] / df["number_of_genes_in_background"].replace(0, np.nan)
    df["-log10(FDR)"] = _neglog10_fdr(df["fdr"])
    label_col = "description" if "description" in df.columns else "term"
    return _enrichment_dotplot(
        df, term_col=label_col, x_col="Gene Ratio", size_col="number_of_genes", color_col="-log10(FDR)",
        x_label="Gene Ratio", title="STRING Enrichment Dot Plot",
        size_legend_label="Gene Count", color_legend_label="-log10(FDR)",
        top_n=top_n, reference_x=None, select_col="fdr", select_ascending=True,
    )
