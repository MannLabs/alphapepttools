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


# Automatically define __all__ to contain public names
__all__ = [name for name in globals() if not name.startswith("_")]
