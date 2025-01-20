from anndata import AnnData


def basic_preproc(adata: AnnData) -> int:
    """Run a basic preprocessing on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    del adata  # unused
    print("Implement a preprocessing function here.")
    return 0
