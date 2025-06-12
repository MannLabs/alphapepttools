from collections.abc import Generator

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphatools.io import read_pg_matrix


@pytest.fixture(
    params=[
        # Set index=True
        {"feature_name": "pg", "sample_name": "sample_id", "set_index": True, "row_type": "features"},
        # # Set index=True and custom sample/feature name
        {"feature_name": "protein_group", "sample_name": "sample_name", "set_index": True, "row_type": "features"},
        # Set index=False
        {"feature_name": "protein_group", "sample_name": "sample_name", "set_index": False, "row_type": "features"},
        # # Transposed matrix (observation/features)
        {"feature_name": "protein_group", "sample_name": "sample_name", "set_index": True, "row_type": "observations"},
        # Set index=False
        {"feature_name": "protein_group", "sample_name": "sample_name", "set_index": False, "row_type": "observations"},
    ]
)
def pg_matrix(tmpdir, request) -> Generator[tuple[str, ad.AnnData, dict[str, str]], None, None]:
    """Generate protein group matrix, reference :class:`anndata.AnnData` object for various argument combinations"""
    tmppath = tmpdir / "pg.tsv"

    feature_name = request.param["feature_name"]
    sample_name = request.param["sample_name"]
    set_index = request.param["set_index"]
    row_type = request.param["row_type"]

    # Generate a obs x features matrix
    X = np.arange(12).reshape(3, 4)
    index = ["A", "B", "C"]
    columns = ["P1", "P2", "P3", "P4"]

    # Simulated protein group matrix of shape
    df = pd.DataFrame(X, index=index, columns=columns)

    if row_type == "features":
        df = df.T

    df.to_csv(tmppath, sep="\t", index=True)

    # anndata sets string numerics "0", "1", ... as index per default
    obs_index = index if set_index else list(map(str, range(len(index))))
    obs = pd.DataFrame(index, index=obs_index, columns=[sample_name])

    var_index = columns if set_index else list(map(str, range(len(columns))))
    var = pd.DataFrame(columns, index=var_index, columns=[feature_name])

    # AnnData stores observations x features
    adata = ad.AnnData(X=X, obs=obs, var=var)

    yield (
        tmppath,
        adata,
        {"feature_name": feature_name, "sample_name": sample_name, "set_index": set_index, "row_type": row_type},
    )

    tmppath.unlink()


def test_read_pg_matrix(pg_matrix) -> None:
    """Test correct parsing of locally stored protein group matrices"""
    path, adata_ref, kwargs = pg_matrix

    adata = read_pg_matrix(path, **kwargs)

    assert (adata_ref.X == adata.X).all()
    assert adata.obs.equals(adata_ref.obs)
    assert adata.var.equals(adata_ref.var)
