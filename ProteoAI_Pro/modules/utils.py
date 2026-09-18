"""
utils.py - Core data loading, validation, and helper utilities for ProteoAI Pro.

Expected input formats
-----------------------
Protein intensity matrix (CSV/XLSX):
    Protein, Gene, Sample1, Sample2, QC1, QC2, ...
    (rows = proteins/features, columns = samples)
    The second column, "Gene" (or a close variant -- see GENE_COLUMN_ALIASES
    below), is OPTIONAL but recommended: a Protein -> Gene SYMBOL mapping used by
    the GSEA and P-P Interaction tabs, since STRING/Enrichr/MSigDB gene sets all
    key on gene symbol rather than raw protein accessions. If omitted, those tabs
    fall back to using the Protein column directly (works only if your Protein
    identifiers already are gene symbols).

Metadata table (CSV/XLSX):
    Sample, Group, IsQC, Batch
    - Sample   : must match a column name in the protein intensity matrix
    - Group    : experimental group / condition label
    - IsQC     : True/False (or 1/0) flag for QC samples
    - Batch    : optional batch identifier (used for batch-specific normalization)
"""

import io
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap


REQUIRED_METADATA_COLS = ["Sample", "Group"]
GENE_COLUMN_ALIASES = {"gene", "gene name", "gene_name", "genesymbol", "gene symbol",
                        "gene names", "genes"}


class DataValidationError(Exception):
    pass


def load_table(file_or_buffer, filename_hint: str = "") -> pd.DataFrame:
    """
    Load a CSV or Excel file into a DataFrame, auto-detecting format and, for CSV,
    auto-detecting text encoding. Many lab instrument/software exports (Excel "CSV"
    saves, older Windows tools) use Windows-1252/Latin-1, not UTF-8 -- e.g. '±' (as
    in 'mean ± SD'), 'µ' (micro), or curly quotes are classic culprits that raise
    UnicodeDecodeError under pandas' UTF-8 default. Falls back through common
    encodings in order; Latin-1 can decode any byte sequence, so this never raises
    UnicodeDecodeError itself (a genuinely corrupt/binary file will instead fail
    validate_peak_matrix/validate_metadata with a clearer structural error).
    """
    name = filename_hint.lower() if filename_hint else getattr(file_or_buffer, "name", "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file_or_buffer)

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            if hasattr(file_or_buffer, "seek"):
                file_or_buffer.seek(0)
            return pd.read_csv(file_or_buffer, encoding=encoding)
        except UnicodeDecodeError:
            continue
    # Unreachable in practice (latin-1 accepts every byte value 0-255), but keeps
    # the function's contract honest if that ever changes.
    if hasattr(file_or_buffer, "seek"):
        file_or_buffer.seek(0)
    return pd.read_csv(file_or_buffer)


def validate_peak_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate raw protein intensity matrix. First column must be protein
    identifiers. An optional SECOND column named 'Gene' (or a close variant, see
    GENE_COLUMN_ALIASES) is recognized as a Protein -> Gene SYMBOL mapping and
    stashed in the returned DataFrame's `.attrs['gene_map']` (a {Protein: Gene}
    dict) rather than kept as a data column -- every downstream numeric operation
    on the matrix (cleaning, QC, normalization, stats, PCA, heatmap, ...) sees
    only the sample columns, exactly as before Gene-column support was added.
    """
    if df.shape[1] < 2:
        raise DataValidationError("Protein intensity matrix must have a protein column plus at least one sample column.")
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Protein"})
    df["Protein"] = df["Protein"].astype(str)
    if df["Protein"].duplicated().any():
        dupes = df["Protein"][df["Protein"].duplicated()].unique().tolist()
        raise DataValidationError(f"Duplicate protein names found: {dupes[:5]}...")

    gene_map = None
    if df.shape[1] >= 3 and str(df.columns[1]).strip().lower() in GENE_COLUMN_ALIASES:
        gene_col = df.columns[1]
        gene_map = dict(zip(df["Protein"], df[gene_col].astype(str)))
        df = df.drop(columns=[gene_col])

    sample_cols = df.columns[1:]
    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df.set_index("Protein")
    if gene_map is not None:
        out.attrs["gene_map"] = gene_map
    return out


def validate_metadata(meta: pd.DataFrame, sample_cols) -> pd.DataFrame:
    """Validate metadata table and align it to the sample columns present in the peak matrix."""
    missing = [c for c in REQUIRED_METADATA_COLS if c not in meta.columns]
    if missing:
        raise DataValidationError(f"Metadata missing required column(s): {missing}")
    if "IsQC" not in meta.columns:
        meta["IsQC"] = False
    else:
        meta["IsQC"] = meta["IsQC"].astype(str).str.lower().isin(["true", "1", "yes", "qc"])
    if "Batch" not in meta.columns:
        meta["Batch"] = "1"
    meta = meta.set_index("Sample")
    missing_samples = [s for s in sample_cols if s not in meta.index]
    if missing_samples:
        raise DataValidationError(f"Samples present in peak matrix but missing from metadata: {missing_samples}")
    return meta.loc[list(sample_cols)]


def split_qc_and_samples(peak_df: pd.DataFrame, meta: pd.DataFrame):
    """Return (qc_columns, sample_columns) based on metadata IsQC flag."""
    qc_cols = meta.index[meta["IsQC"]].tolist()
    sample_cols = meta.index[~meta["IsQC"]].tolist()
    return qc_cols, sample_cols


def get_categorical_metadata_columns(meta: pd.DataFrame, sample_cols=None, max_unique: int = 15) -> list:
    """
    Which metadata columns are usable as a grouping/coloring variable (for PCA,
    Statistics, Heatmap column annotation, Boxplot) -- not just the hardcoded
    'Group' column. A column qualifies if, among biological samples only (QC rows'
    placeholder values like 'QC'/NaN are excluded from this check), it's non-numeric
    (object/category dtype -- covers text labels like Diagnosis, Gender, Treatment,
    Ethnicity) or numeric with few enough distinct values to plausibly be a
    group/category rather than a continuous measurement (excludes Age, Body Weight,
    and similar continuous covariates, which aren't meaningful t-test/ANOVA/boxplot
    groups). 'Group' is always included first if present, for a stable default.
    """
    if sample_cols is None:
        sample_cols = meta.index.tolist()
    bio_meta = meta.loc[meta.index.intersection(sample_cols)]
    candidates = []
    for col in meta.columns:
        if col in ("IsQC",):
            continue
        series = bio_meta[col].dropna()
        if series.empty:
            continue
        n_unique = series.nunique()
        if n_unique < 2 or n_unique > len(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= max_unique:
                candidates.append(col)
        else:
            candidates.append(col)
    # stable, predictable ordering: Group first (if present), then the rest as-authored
    ordered = [c for c in ("Group",) if c in candidates]
    ordered += [c for c in candidates if c not in ordered]
    return ordered


def zero_replacement(df: pd.DataFrame, method: str = "min_fraction", fraction: float = 0.5) -> pd.DataFrame:
    """
    Replace zeros / missing values prior to log transform.
    method:
      - 'min_fraction': replace with fraction * (smallest non-zero value in that feature's row)
      - 'global_min': replace with fraction * (smallest non-zero value across whole matrix)
    """
    out = df.copy()
    if method == "min_fraction":
        for idx in out.index:
            row = out.loc[idx]
            nonzero = row[(row > 0) & row.notna()]
            fill_val = nonzero.min() * fraction if len(nonzero) else np.nan
            out.loc[idx] = row.where((row > 0) & row.notna(), fill_val)
    else:  # global_min
        nonzero = out.values[(out.values > 0) & (~np.isnan(out.values))]
        fill_val = nonzero.min() * fraction if nonzero.size else np.nan
        out = out.where((out > 0) & out.notna(), fill_val)
    return out


def to_download_bytes_csv(df: pd.DataFrame) -> bytes:
    """
    Serialize a DataFrame to CSV bytes for st.download_button, with a UTF-8 BOM
    (utf-8-sig). Protein names routinely contain Greek letters (α, β, Δ),
    symbols (±, µ, °), or accented Latin characters -- plain 'utf-8' without a BOM
    is valid and round-trips fine in Python/pandas, but Excel (still the most common
    tool users re-open these exports in) assumes the system locale encoding for a
    plain UTF-8 CSV and renders those characters as mojibake unless a BOM marks it
    explicitly as UTF-8. The BOM is a no-op for pandas/any UTF-8-aware reader.
    """
    return df.to_csv().encode("utf-8-sig")


def to_download_bytes_xlsx(sheets: dict) -> bytes:
    """sheets: dict of {sheet_name: DataFrame}"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, d in sheets.items():
            safe_name = name[:31]
            d.to_excel(writer, sheet_name=safe_name)
    buf.seek(0)
    return buf.read()


def build_colormap(base_cmap: str = "RdBu_r", custom_colors=None, reverse: bool = False):
    """
    Build a matplotlib colormap either from a predefined name or a custom list of hex
    colors picked via a 2D color picker (creates a smooth gradient through them via
    LinearSegmentedColormap). Shared by the Heatmap and P-P Interaction tabs so both
    offer the same "predefined palette OR pick your own colors" experience.
    """
    if custom_colors:
        cmap = LinearSegmentedColormap.from_list("custom_gradient", custom_colors, N=256)
    else:
        cmap = matplotlib.colormaps.get_cmap(base_cmap)
    if reverse:
        cmap = cmap.reversed()
    return cmap


# ---------------------------------------------------------------------------
# Figure export helpers (shared by every tab that renders a downloadable plot)
# ---------------------------------------------------------------------------
def fig_to_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    """
    Serialize a matplotlib figure to bytes in the given format, for
    st.download_button. Supports 'png'/'tiff'/'jpg' (raster, dpi matters) and
    'svg'/'pdf' (vector, dpi is ignored by matplotlib but harmless to pass).
    """
    buf = io.BytesIO()
    save_fmt = "jpeg" if fmt == "jpg" else fmt
    fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


FIGURE_EXPORT_FORMATS = {
    "PNG (raster)": ("png", "image/png"),
    "TIFF (raster, publication)": ("tiff", "image/tiff"),
    "SVG (vector)": ("svg", "image/svg+xml"),
    "PDF (vector)": ("pdf", "application/pdf"),
}


# On-screen rendering DPI for every matplotlib figure shown via render_figure()/
# render_figure_grid() below -- independent of download resolution (which the user
# picks separately in render_figure_download()). Streamlit's st.pyplot() defaults to
# width='stretch', so a figure placed in a narrower st.columns() cell is shrunk to
# fit; without a high enough source DPI, that downscaling (combined with a HiDPI/
# retina screen rendering the shrunk image back up) is exactly what produces the
# blurry/hazy look this DPI raises to prevent.
ON_SCREEN_FIGURE_DPI = 160


def format_df_decimals(df: pd.DataFrame, decimals: int = 6):
    """
    DISPLAY-ONLY formatting for a table shown via st.dataframe(): every numeric
    value is shown with up to `decimals` decimal places (trailing zeros
    trimmed, so 0.5 stays "0.5" rather than "0.500000"). Any nonzero value
    that would round to 0 at that precision (e.g. a very small p-value or FDR)
    is shown in scientific notation instead, so it never gets misread as
    exactly zero.

    Returns a NEW DataFrame with formatted values as strings -- the original
    df (and anything derived from it for calculations or CSV downloads) is
    left completely untouched; only pass the *result* of this function to
    st.dataframe(), never to a download/export or a further calculation.
    Integers, booleans, text, and missing values pass through unchanged.
    """
    if df is None or len(df) == 0:
        return df

    threshold = 10 ** (-decimals)

    def _fmt(x):
        if isinstance(x, (bool, np.bool_)):
            return x
        try:
            if pd.isna(x):
                return x
        except (TypeError, ValueError):
            pass
        if isinstance(x, (int, np.integer)):
            return x
        if isinstance(x, (float, np.floating)):
            if x == 0:
                return "0"
            if abs(x) < threshold or abs(x) >= 1e9:
                return f"{x:.{decimals}e}"
            s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
            return s if s not in ("", "-") else "0"
        return x

    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_fmt)
    return out


def render_figure(st_module, fig, dpi: int = ON_SCREEN_FIGURE_DPI):
    """
    Display one matplotlib figure at a crisp, consistent on-screen resolution.
    Sets the figure's own DPI (matplotlib's default is 100) before handing it to
    st.pyplot(), so text, axis labels, legends, and annotations stay sharp even
    when the figure is displayed at reduced width (e.g. inside a column) or on a
    high-density display. Purely a display-quality change -- it does not touch
    the figure's data, size in inches, or any computed values.
    """
    try:
        fig.set_dpi(dpi)
    except Exception:
        pass
    st_module.pyplot(fig, width="stretch")


def render_figure_grid(st_module, panels: list, dpi: int = ON_SCREEN_FIGURE_DPI):
    """
    Lay out a set of related figures side-by-side instead of stacked vertically,
    using the screen width efficiently while keeping each figure legible:
      1 figure  -> full width
      2 figures -> 2 columns
      3 figures -> 3 columns
      4 figures -> 2x2 grid
      >4        -> wraps into further rows of up to 3

    Each entry in `panels` is a dict describing ONE figure and everything that
    goes with it, so nothing is dropped when the layout changes:
      fig            (required)  the matplotlib Figure
      title          (optional)  markdown header shown above the figure
      caption        (optional)  st.caption text shown above the figure
      download_name  (optional)  base filename for the download row
      download_key   (required if download_name is set) unique widget key prefix

    This only changes arrangement/resolution -- every figure, title, caption, and
    download button that existed before is still rendered, exactly as before.
    """
    def _draw(container, spec):
        if spec.get("title"):
            container.markdown(spec["title"])
        if spec.get("caption"):
            container.caption(spec["caption"])
        render_figure(container, spec["fig"], dpi=dpi)
        if spec.get("download_name"):
            render_figure_download(container, spec["fig"], spec["download_name"],
                                    key_prefix=spec["download_key"])

    n = len(panels)
    if n == 0:
        return
    if n == 1:
        _draw(st_module, panels[0])
        return
    if n == 4:
        rows = [panels[0:2], panels[2:4]]
    else:
        chunk = min(n, 3)
        rows = [panels[i:i + chunk] for i in range(0, n, chunk)]
    for row in rows:
        cols = st_module.columns(len(row))
        for spec, col in zip(row, cols):
            _draw(col, spec)


def align_download_button(container):
    """
    st.selectbox / st.number_input render a visible label line above their input
    box; st.download_button has no such label line, so when placed in a column
    next to a selectbox it renders noticeably higher -- the button and the
    selectbox's actual input boxes end up on different baselines. This inserts
    an INVISIBLE label-sized spacer (same font-size/line-height/margin as a real
    Streamlit widget label, just hidden) immediately before the button, so its
    top edge lines up with the selectbox input box beside it rather than the
    selectbox's label. Used everywhere a Format/DPI selector sits next to a
    download button, so every export row in the app aligns the same way.
    """
    container.markdown(
        "<div style='visibility:hidden; font-size:14px; line-height:1.6; "
        "margin-bottom:0.25rem;'>&nbsp;</div>",
        unsafe_allow_html=True,
    )


def render_figure_download(st_module, fig, base_filename: str, key_prefix: str,
                            dpi_options=(150, 300, 600), default_dpi_index: int = 1):
    """
    Shared 'download this figure' widget: format selector (PNG/TIFF/SVG/PDF) + DPI
    selector (for raster formats) + a download button, all in one row, with the
    button's top edge aligned to the selectors' input boxes (see
    align_download_button() above). Used everywhere a plot needs a high-resolution,
    multi-format download (QC plots, PCA plots) so the control looks and behaves
    identically across tabs.
    `key_prefix` must be unique per call site to avoid Streamlit widget-key collisions.
    """
    c1, c2, c3 = st_module.columns([2, 1, 1.4])
    fmt_label = c1.selectbox(
        "Format", list(FIGURE_EXPORT_FORMATS.keys()), key=f"{key_prefix}_fmt", index=0
    )
    fmt, mime = FIGURE_EXPORT_FORMATS[fmt_label]
    is_raster = fmt in ("png", "tiff", "jpg")
    if is_raster:
        dpi = c2.selectbox("DPI", list(dpi_options), key=f"{key_prefix}_dpi",
                            index=default_dpi_index)
    else:
        dpi = 300
        c2.caption("(vector — resolution-independent)")
    file_name = f"{base_filename}.{fmt}"
    align_download_button(c3)
    c3.download_button(
        f"Download {fmt.upper()}", fig_to_bytes(fig, fmt=fmt, dpi=dpi),
        file_name=file_name, mime=mime, key=f"{key_prefix}_dl"
    )
