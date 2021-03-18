from numbers import Number
from typing import List, Optional, Tuple, Union

from matplotlib.colors import Colormap, Normalize
from numpy import ndarray
import pandas as pd
from scipy.sparse import spmatrix
import seaborn as sns
from seaborn import JointGrid
from sklearn.base import BaseEstimator


def jointplot(
    data: Union[pd.DataFrame, ndarray, spmatrix],
    embedding_transformer: BaseEstimator,
    kind: str = "scatter",
    color: Optional[Union[str, Colormap]] = None,
    height: Number = 6,
    ratio: Number = 5,
    space: Number = 0.2,
    dropna: bool = False,
    xlim: Optional[Number] = None,
    ylim: Optional[Number] = None,
    marginal_ticks: bool = False,
    joint_kws: Optional[dict] = None,
    marginal_kws: Optional[dict] = None,
    hue: Optional[Union[str, list, ndarray]] = None,
    palette: Optional[Union[str, list, dict, Colormap]] = None,
    hue_order: Optional[List[str]] = None,
    hue_norm: Optional[Union[tuple, Normalize]] = None,
    **kwargs
) -> Tuple[JointGrid, pd.DataFrame]:
    if isinstance(data, pd.DataFrame):
        X = data.values
    else:
        X = data
    X_embedding: ndarray = embedding_transformer.fit_transform(X)
    df_embedding: pd.DataFrame = pd.DataFrame(
        data=X_embedding, index=data.index, columns=["x1", "x2"]
    )
    return (
        sns.jointplot(
            data=df_embedding,
            x="x1",
            y="x2",
            kind=kind,
            color=color,
            height=height,
            ratio=ratio,
            space=space,
            dropna=dropna,
            xlim=xlim,
            ylim=ylim,
            marginal_ticks=marginal_ticks,
            joint_kws=joint_kws,
            marginal_kws=marginal_kws,
            hue=hue,
            palette=palette,
            hue_order=hue_order,
            hue_norm=hue_norm,
            **kwargs,
        ),
        df_embedding,
    )
