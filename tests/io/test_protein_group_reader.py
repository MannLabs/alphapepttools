from collections.abc import Generator

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.io import read_pg_matrix


@pytest.fixture(
    params=[
        # Basic
        {
            "feature_name": ("pg",),
            "sample_name": ("sample_id",),
            "row_type": "features",
            "sample_metadata_index": 0,
            "feature_metadata_index": 0,
            "sample_metadata": (("A",), ("B",), ("C",)),
            "feature_metadata": (("P1",), ("P2",), ("P3",), ("P4",)),
        },
        # Custom sample/feature name
        {
            "feature_name": ("protein_group",),
            "sample_name": ("sample_name",),
            "row_type": "features",
            "sample_metadata_index": 0,
            "feature_metadata_index": 0,
            "sample_metadata": (("A",), ("B",), ("C",)),
            "feature_metadata": (("P1",), ("P2",), ("P3",), ("P4",)),
        },
        # Transposed matrix (observation/features)
        {
            "feature_name": ("protein_group",),
            "sample_name": ("sample_name",),
            "row_type": "observations",
            "sample_metadata_index": 0,
            "feature_metadata_index": 0,
            "sample_metadata": (("A",), ("B",), ("C",)),
            "feature_metadata": (("P1",), ("P2",), ("P3",), ("P4",)),
        },
        ## Additional metadata ##
        # Only features
        {
            "feature_name": ("protein_id", "seq"),
            "sample_name": ("sample_id",),
            "row_type": "features",
            "sample_metadata_index": [0],
            "feature_metadata_index": [0, 1],
            "sample_metadata": (("A",), ("B",), ("C",)),
            "feature_metadata": (("P1", "AAA"), ("P2", "CCC"), ("P3", "DDD"), ("P4", "EEE")),
        },
        # Only samples
        {
            "feature_name": ("protein_id",),
            "sample_name": ("sample_id", "patient_id"),
            "row_type": "features",
            "sample_metadata_index": [0, 1],
            "feature_metadata_index": [0],
            "sample_metadata": (("A", "p1"), ("B", "p2"), ("C", "p3")),
            "feature_metadata": (("P1",), ("P2",), ("P3",), ("P4",)),
        },
        {
            "feature_name": ("protein_id", "seq"),
            "sample_name": ("sample_id", "patient_id"),
            "row_type": "features",
            "sample_metadata_index": [0, 1],
            "feature_metadata_index": [0, 1],
            "sample_metadata": (("A", "p1"), ("B", "p2"), ("C", "p3")),
            "feature_metadata": (("P1", "AAA"), ("P2", "CCC"), ("P3", "DDD"), ("P4", "EEE")),
        },
    ]
)
def pg_matrix(tmpdir, request) -> Generator[tuple[str, ad.AnnData, dict[str, str]], None, None]:
    """Generate protein group matrix, reference :class:`anndata.AnnData` object for various argument combinations"""
    tmppath = tmpdir / "pg.tsv"

    feature_metadata_index = request.param["feature_metadata_index"]
    sample_metadata_index = request.param["sample_metadata_index"]
    feature_name = request.param["feature_name"]
    sample_name = request.param["sample_name"]
    row_type = request.param["row_type"]
    index = request.param["sample_metadata"]
    columns = request.param["feature_metadata"]

    # Generate a obs x features matrix
    X = np.arange(12).reshape(3, 4)

    # Simulated protein group matrix of shape
    df = pd.DataFrame(
        X,
        index=pd.MultiIndex.from_tuples(index, names=sample_name),
        columns=pd.MultiIndex.from_tuples(columns, names=feature_name),
    )

    if row_type == "features":
        df = df.T

    # Pandas writes index names as additional row, which does
    # not correspond to the expected format of PG matrices.
    # Make indices actual columns to prevent this
    df.reset_index().to_csv(tmppath, sep="\t", index=False)

    # anndata sets string numerics "0", "1", ... as index per default
    obs_index = list(map(str, range(len(index))))
    # obs_columns = pd.MultiIndex.from_tuples(index, names=sample_name)
    obs = pd.DataFrame(index, index=obs_index, columns=sample_name)

    var_index = list(map(str, range(len(columns))))
    # var_columns = pd.MultiIndex.from_tuples(index, names=feature_name)
    var = pd.DataFrame(columns, index=var_index, columns=feature_name)

    # AnnData stores observations x features
    adata = ad.AnnData(X=X, obs=obs, var=var)

    sample_name = sample_name[0] if len(sample_name) == 1 else sample_name
    feature_name = feature_name[0] if len(feature_name) == 1 else feature_name

    yield (
        tmppath,
        adata,
        {
            "row_type": row_type,
            "feature_metadata_index": feature_metadata_index,
            "sample_metadata_index": sample_metadata_index,
            "feature_name": feature_name,
            "sample_name": sample_name,
        },
    )

    tmppath.remove()


def test_read_pg_matrix(pg_matrix) -> None:
    """Test correct parsing of locally stored protein group matrices"""
    path, adata_ref, kwargs = pg_matrix

    adata = read_pg_matrix(path, **kwargs)

    assert (adata_ref.X == adata.X).all()
    assert adata.obs.equals(adata_ref.obs)
    assert adata.var.equals(adata_ref.var)
