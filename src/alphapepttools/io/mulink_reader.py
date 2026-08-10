# Reader wrapper to generate MuLink (https://github.com/lucas-diedrich/mulink) instances from reports

import importlib

import anndata as ad
import mudata as md
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from alphapepttools.io.reader_columns import ALPHAPEPTTOOLS_FEATURE_ID_NAME, FEATURE_LEVEL_CONFIG

# Port from https://github.com/lucas-diedrich/mulink/blob/main/docs/notebooks/protein-data.ipynb?short_path=4fbb043, with adjustments to build a MuLink instance from a set of AnnData objects rather than a PSM-table


def _sparse_matrix_mapping(
    mapping_df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    transitive_closure: bool = True,
) -> pd.DataFrame:
    """Create a square sparse adjacency matrix for varp from a vertix mapping.

    Parameters
    ----------
    mapping_df : pd.DataFrame
        DataFrame encoding the edges between source vertices and target vertices.
    kwargs
        Passed to :func:`nx.from_pandas_edgelist`

    Returns
    -------
    Dataframe representing the adjacency matrix of the respective directed acylic graph.
    Directed graph in which source vertices are pointing towards target vertices.
    The semantics of the edges depend on the application
    """
    column_names = mapping_df.columns.tolist() if columns is None else columns

    full_dag = nx.DiGraph()
    for idx in tqdm(range(len(column_names) - 1)):
        dag = nx.from_pandas_edgelist(
            mapping_df, source=column_names[idx], target=column_names[idx + 1], create_using=nx.DiGraph
        )
        full_dag = nx.compose(full_dag, dag)

    if transitive_closure:
        full_dag = nx.transitive_closure_dag(full_dag)

    return pd.DataFrame.sparse.from_spmatrix(
        nx.to_scipy_sparse_array(full_dag), index=full_dag.nodes, columns=full_dag.nodes
    )


def _reindex_adjacency_matrix(df: pd.DataFrame, new_index: pd.Index) -> csr_matrix:
    """Reindex a sparse adjacency-matrix DataFrame to `new_index` along both axes

    Labels in `new_index` not present in `df.index` become all-zero
    rows/columns (implicit, costing no memory). Labels in `df.index` not
    present in `new_index` are dropped.

    Assumes `df.index.equals(df.columns)` and unique labels.
    """
    if not df.index.equals(df.columns):
        raise ValueError("Adjacency DataFrame must have matching index and columns")
    if not df.index.is_unique:
        raise ValueError("Adjacency index must be unique")

    # Map each position in new_index to a position in df.index (-1 if missing).
    indexer = df.index.get_indexer(new_index)
    keep = indexer >= 0
    src = indexer[keep]  # positions to take from the original
    dst = np.flatnonzero(keep)  # where they land in the new matrix

    # Pull out the submatrix corresponding to nodes that survive.
    # CSR for row slicing, then CSC for column slicing — each is O(nnz of
    # the selected slice) rather than scanning the whole matrix.
    mat = df.sparse.to_coo().tocsr()
    sub = mat[src, :].tocsc()[:, src].tocoo()

    # Remap the submatrix's local (row, col) into positions in new_index.
    n = len(new_index)

    return csr_matrix(
        (sub.data, (dst[sub.row], dst[sub.col])),
        shape=(n, n),
        dtype=mat.dtype,
    )


def mulink_from_anndatas(
    anndatas: dict[str, ad.AnnData],
    *,
    transitive_closure: bool = True,
) -> md.MuData:
    """Build a :class:`mudata.MuData` with cross-level feature linkage from a set of AnnData objects.

    Combines one AnnData per feature level (genes, proteins, peptides, precursors) into a single
    MuData and attaches a sparse adjacency matrix to ``.varp["feature_mapping"]`` that links
    features across levels. The resulting object can be queried with :mod:`link` to filter the
    hierarchy by ancestors or descendants of a set of features (see Examples).

    The cross-level mapping is reconstructed entirely from each AnnData's ``.var``: each
    non-coarsest level must carry the higher, i.e. coarser level's feature-id column. When the AnnData
    objects come from :func:`alphapepttools.io.read_psm_table`, this is controlled by the
    ``var_columns`` argument of that function. The original ``.var``, ``.obs``, and ``.X`` of
    each input AnnData are preserved unchanged on the corresponding modality of the returned
    MuData.

    Parameters
    ----------
    anndatas
        Dictionary mapping feature levels to AnnData objects. Keys must be a subset of
        ``{"genes", "proteins", "peptides", "precursors"}``. Each AnnData's ``var`` index name
        must equal the level's feature-id column (e.g. ``"proteins"`` for the proteins level),
        and each non-coarsest level must include the next-coarser level's id as a column in
        ``.var``, e.g. `precursors.var` must contain a column named `proteins` if it is to be linked to
        a protein-level AnnData object.
    transitive_closure
        If True (default), the linkage matrix encodes all reachable (u, v) pairs across levels,
        so e.g. precursors are directly linked to genes without going through peptides or
        proteins. If False, only direct (adjacent-level) edges are stored.

    Returns
    -------
    :class:`mudata.MuData`
        MuData with one modality per input level and the cross-level adjacency matrix attached
        at ``.varp["feature_mapping"]``.

    Examples
    --------
    Read precursor-, protein-, and gene-level AnnData objects and assemble them into a MuLink
    instance. The ``var_columns`` argument is what makes the levels linkable: each level needs
    the next-coarser level's id as a column in ``.var``.

    .. code-block:: python

        import alphapepttools as apt

        data_path = apt.data.get_data("bader2020_psm_diann")

        adata_precursor = apt.io.read_psm_table(
            file_paths=data_path / "top20_report.parquet",
            search_engine="diann",
            level="precursors",
            var_columns=["proteins", "genes"],
        )

        adata_protein = apt.io.read_psm_table(
            file_paths=data_path / "top20_report.parquet",
            search_engine="diann",
            level="proteins",
            var_columns="genes",
        )

        mlink = apt.io.mulink_from_anndatas(
            anndatas={
                "precursors": adata_precursor,
                "proteins": adata_protein,
            }
        )

    Filter the MuData down to features linked to a set of genes via the link query accessor.
    Each ``.mod[level]`` is the original AnnData restricted to the matched features, with
    ``.obs`` and ``.var`` intact:

    .. code-block:: python

        filtered = mlink.link.query.ancestors(["TF", "SERPINA3"])
        filtered_proteins = filtered.mod["proteins"]
        filtered_precursors = filtered.mod["precursors"]

    """
    available_feature_levels = FEATURE_LEVEL_CONFIG.keys()

    # validate feature levels against alphapepttools reference config
    for feature_level in anndatas:
        if feature_level not in FEATURE_LEVEL_CONFIG:
            raise ValueError(
                f"Invalid feature level {feature_level}. Must be one of {list(FEATURE_LEVEL_CONFIG.keys())}"
            )

    # validate that the index name equals the feature ID column name
    for feature_level, adata in anndatas.items():
        feature_id_column = FEATURE_LEVEL_CONFIG[feature_level][ALPHAPEPTTOOLS_FEATURE_ID_NAME]
        if adata.var.index.name != feature_id_column:
            raise ValueError(
                f"Anndata for feature level {feature_level!r} has index name {adata.var.index.name!r} but expected "
                f"{feature_id_column!r} based on the reference PSM table config. Please ensure that the anndata "
                f"objects were created from the same reference PSM table and that the index name was not modified."
            )

    # inference of hierarchy without psm table & enforcing correct order
    actual_feature_levels = [feature_level for feature_level in available_feature_levels if feature_level in anndatas]

    # For each non-coarsest included level get a 2-column [own_id, next_coarser_id] frame.
    # own_id comes from the (named) index. It's possible that levels are skipped, so long as
    # the next coarser level's id column is present in the var of the finer level.
    pairwise_mappings: dict[str, pd.DataFrame] = {}
    for idx, current_level in enumerate(actual_feature_levels):
        # coarsest included level has no coarser link to bring in
        if idx == 0:
            continue

        higher_level = actual_feature_levels[idx - 1]
        higher_level_feature_id_column = FEATURE_LEVEL_CONFIG[higher_level][ALPHAPEPTTOOLS_FEATURE_ID_NAME]
        var = anndatas[current_level].var
        if higher_level_feature_id_column not in var.columns:
            raise ValueError(
                f"Anndata for feature level {current_level!r} is missing column "
                f"{higher_level_feature_id_column!r} required to link to next included coarser level {higher_level!r}. "
                f"Try running alphepeptools.io.read_psm_table(..., var_columns={higher_level!r}) to include the necessary column in the anndata object."
            )
        pairwise_mappings[current_level] = var[[higher_level_feature_id_column]].reset_index()

    # Walk from finest to coarsest and chain-merge each pairwise mapping on its own_id
    # column (which is also the next-coarser-id column of the previously-merged finer
    # level). Non-adjacent (transitive) edges across skipped levels are recovered
    # downstream by sparse_matrix_mapping via the DAG transitive closure.
    levels_fine_to_coarse = list(reversed(actual_feature_levels))
    finest_level = levels_fine_to_coarse[0]
    mapping_df = pairwise_mappings[finest_level].copy()
    for current_level in levels_fine_to_coarse[1:]:
        if current_level not in pairwise_mappings:
            # coarsest included level: column already pulled in via the previous-finer merge
            continue
        right = pairwise_mappings[current_level]
        shared_col = right.columns[0]  # own_id of current_level
        mapping_df = mapping_df.merge(right, on=shared_col, how="left")

    adjancency_matrix = _sparse_matrix_mapping(
        mapping_df,
        columns=[FEATURE_LEVEL_CONFIG[level][ALPHAPEPTTOOLS_FEATURE_ID_NAME] for level in levels_fine_to_coarse],
        transitive_closure=transitive_closure,
    )

    # pull_on_update=False adopts mudata 0.4 behavior: mdata.var stays clean (just
    # modality/feature membership), without merging per-modality var columns across
    # incompatible feature levels.
    with md.set_options(pull_on_update=False):
        mdata = md.MuData(data=anndatas)

    # IMPORTANT: features in the adjacency matrix need to be aligned with the variable order in the mdata object
    adjancency_matrix = _reindex_adjacency_matrix(adjancency_matrix, new_index=mdata.var_names)

    # Import mulink lazily here to avoid the global side-effect
    importlib.import_module("mulink")
    mdata.link.add_link(adjancency_matrix)

    return mdata
