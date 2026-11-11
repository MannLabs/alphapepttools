"""Package-wide settings"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

_TRACK_PROVENANCE_DEFAULT = False
_track_provenance: ContextVar[bool] = ContextVar("track_provenance", default=_TRACK_PROVENANCE_DEFAULT)


class Settings:
    """Runtime settings for alphapepttools

    Attributes are read at call time by the functions that use them. :meth:`override` changes settings for a block of code and restores them afterwards, e.g. to skip provenance tracking temporarily
    """

    @property
    def track_provenance(self) -> bool:
        """Whether functions with the provenance tracking decorator record an entry in `adata.uns["provenance"]`."""
        return _track_provenance.get()

    @track_provenance.setter
    def track_provenance(self, value: bool) -> None:
        _track_provenance.set(value)

    def reset_track_provenance(self) -> None:
        """Restore provenance tracking to package default"""
        _track_provenance.set(_TRACK_PROVENANCE_DEFAULT)

    @contextmanager
    def override(self, *, track_provenance: bool | None = None) -> Generator[None, None, None]:
        """Temporarily override settings in the enclosed block

        Parameters
        ----------
        track_provenance
            Value for the `track_provenance` attr inside the block, default None (no override).

        Examples
        --------
        .. code-block:: python

            import alphapepttools as apt
            import anndata as ad
            import pandas as pd

            adata = ad.AnnData(
                pd.DataFrame(
                    {
                        "col1": [1, 2, 3],
                        "col2": [4, 5, 6],
                    }
                )
            )

            with apt.settings.override(track_provenance=False):
                apt.pp.normalize(adata)  # not recorded in provenance

            print(adata.uns.get("provenance"))  # no entry for normalize

        """
        token = _track_provenance.set(track_provenance) if track_provenance is not None else None
        try:
            yield
        finally:
            if token is not None:
                # Restore the original value of track_provenance also if the block fails
                _track_provenance.reset(token)


settings = Settings()
