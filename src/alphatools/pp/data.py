# Create and manipulate Anndata objects

import anndata as ad
import numpy as np
import pandas as pd
from typing import Union
import logging

# logging configuration
logging.basicConfig(level=logging.INFO)

def to_anndata(
		data: Union[np.ndarray, pd.DataFrame],
		obs: pd.DataFrame = None,
		var: pd.DataFrame = None,	
		keep_all_data_rows: bool = False,
		keep_all_data_cols: bool = False,
) -> ad.AnnData:
	"""Create an AnnData object from a data array and optional sample & feature metadata
	
	Parameters
	----------
	
	data : np.ndarray or pd.DataFrame
		The data array, where rows correspond to samples and columns correspond to features.
		If data is a pd.DataFrame, row and column indices are used for inner join with obs and var.
		If data is a np.ndarray, obs and var must match the number of rows and columns, respectively.
		
	obs : pd.DataFrame, optional
		Dataframe with sample metadata. If data is a pd.DataFrame, row indices of data and obs must match to be added.

	var : pd.DataFrame, optional
		Dataframe with feature metadata. If data is a pd.DataFrame, column indices of data and var must match to be added.

	keep_all_data_rows : bool, default False
		Whether to keep all rows of data, even if they are not present in obs. Non-matching rows will be set to NaN in obs.

	keep_all_data_cols : bool, default False
		Whether to keep all columns of data, even if they are not present in var. Non-matching columns will be set to NaN in var.

	data_idx_level : int, default 0
		Level of row index to use as obs if data is a pd.DataFrame.

	data_col_level : int, default 0
		Level of column index to use as var if data is a pd.DataFrame.

	obs_idx_level : int, default 0
		Level of row index to use as index of obs.

	var_idx_level : int, default 0
		Level of column index to use as index of var.

	Returns:

	adata : ad.AnnData
		Anndata object with data, obs, and var.
	
	"""

	# If data is a dataframe, convert row and col indices to obs and var
	if isinstance(data, pd.DataFrame):
		adata = ad.AnnData(data)
		adata.obs = data.index.to_frame(name=None)
		adata.var = data.columns.to_frame(name=None)
	elif isinstance(data, np.ndarray):
		adata = ad.AnnData(data)

		# If obs and var are provided, they need to match in shape
		if obs is not None and obs.shape[0] != data.shape[0]:
			logging.info("obs must have the same number of rows as data, skipping...")
			obs = None
		if var is not None and var.shape[0] != data.shape[1]:
			logging.info("var must have the same number of columns as data, skipping...")
			var = None

	else:
		raise TypeError("data must be a pd.DataFrame or np.ndarray")
	
	# If obs is provided, add to adata
	if obs is not None:
		if not isinstance(obs, pd.DataFrame):
			raise TypeError("obs must be a pd.DataFrame")
		adata = add_metadata(
			adata, 
			obs, 
			axis=0, 
			keep_data_shape=keep_all_data_rows,
			keep_existing_metadata=False,
			verbose=False,)
	# If var is provided, add to adata
	if var is not None:
		if not isinstance(var, pd.DataFrame):
			raise TypeError("var must be a pd.DataFrame")
		adata = add_metadata(
			adata,
			var,
			axis=1,
			keep_data_shape=keep_all_data_cols,
			keep_existing_metadata=False,
			verbose=False,)

	return adata

def add_metadata(
		adata: ad.AnnData,
		metadata: pd.DataFrame,
		axis: int,
		keep_data_shape: bool = False,
		keep_existing_metadata: bool = False,
		verbose: bool = True,
) -> ad.AnnData:
	"""Add metadata to an AnnData object while checking for matching indices or shape
	
	If axis is 0, assume metadata.index <-> data.index and add metadata as '.obs' attribute of the AnnData object.
    If axis is 1, assume metadata.index <-> data.columns and add metadata as '.var' attribute of the AnnData object.

	Parameters
	----------
	
	adata : ad.AnnData
		Anndata object to add metadata to.
	
	metadata : pd.DataFrame
		Metadata dataframe to add. The matching entity is always the INDEX, depending on axis it is 
		matched against obs (axis = 0) or var (axis = 1). Correspondingly, the data-aligned dimension
		of both adata.obs and adata.var is always the index.

	"""

	if metadata is None:
		return adata
	
	if not isinstance(metadata, pd.DataFrame):
		raise TypeError("metadata must be a pd.DataFrame")

	# Reject multi-level indices, return to simple
	if metadata.index.nlevels > 1:
		raise ValueError("Multi-level indices are not supported.")
	
	# duplicate metadata indices are not supported
	if any(metadata.index.duplicated()):
		raise ValueError("Duplicated metadata indices are not supported.")
	
	# set join type
	join = "inner" if not keep_data_shape else "left"

	# if existing metadata should be kept and new metadata contains synonymous fields to existing metadata, drop incoming fields
	if keep_existing_metadata:

		if axis == 0:
			_inplace_metadata = adata.obs
		elif axis == 1:
			_inplace_metadata = adata.var

		overlapping_fields = metadata.columns.intersection(_inplace_metadata.columns)
		
		if overlapping_fields.size > 0:
			if verbose:
				logging.info(f"pp.add_metadata(): Synonymous fields, dropping {overlapping_fields.to_list()} (axis = {axis}) from incoming metadata.")
			metadata = metadata.drop(
				adata.obs.columns if axis == 0 else adata.var.columns, 
				axis=1, 
				errors='ignore',
				inplace=False,
			)

		# join new to existing metadata; same join logic as for data
		if verbose:
			logging.info(f"pp.add_metadata(): Join incoming to existing metadata via {join} join on axis  {axis}.")
		metadata = _inplace_metadata.join(metadata, how=join)
		
	# align metadata to data. Based on axis, metadata index will be aligned
	# to either data index (axis = 0) or data columns (axis = 1)
	if axis == 0:
		_df, _md = _get_df_from_adata(adata).align(metadata, axis=0, join=join)
		if not _df.index.equals(_md.index):
			raise ValueError(f"Attempted alignment; metadata indices do not match data indices.")
	elif axis == 1:
		# transpose metadata to align with data columns
		_df, _md = _get_df_from_adata(adata).align(metadata.T, axis=1, join=join)
		if not _df.columns.equals(_md.columns):
			raise ValueError(f"Attempted alignment; metadata columns do not match data columns.")
		
	# new AnnData object
	_adata = ad.AnnData(_df.values)
	_adata.obs = _md if axis == 0 else adata.obs
	_adata.var = _md.T if axis == 1 else adata.var 
	if verbose:
		logging.info(f"pp.add_metadata(): Data {adata.shape} to {_adata.shape}; obs {adata.obs.shape} to {_adata.obs.shape}; var {adata.var.shape} to {_adata.var.shape} \n")

	return _adata

def _get_df_from_adata(
		adata: ad.AnnData,
):
	"""Extract dataframe from AnnData object
	
	Parameters
	----------
	
	adata : ad.AnnData
		Anndata object to extract data from.
		
	Returns
	-------
	
	df : pd.DataFrame
		Dataframe with data from adata.
	
	"""

	out_dataframe = pd.DataFrame(
		adata.X, index = adata.obs.index, columns = adata.var.index
	)

	return out_dataframe

def impute():
	raise NotImplementedError

def scale():
	raise NotImplementedError

# Automatically define __all__ to contain public names
__all__ = [name for name in globals() if not name.startswith("_")]
