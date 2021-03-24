import math
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib.colors import Colormap, Normalize
from numpy import ndarray
from seaborn import JointGrid
from sklearn.base import BaseEstimator

numeric = Union[int, float]


def jointplot(
    data: pd.DataFrame,
    embedding_transformer: BaseEstimator,
    kind: str = "scatter",
    color: Optional[Union[str, Colormap]] = None,
    height: numeric = 6,
    ratio: numeric = 5,
    space: numeric = 0.2,
    dropna: bool = False,
    xlim: Optional[numeric] = None,
    ylim: Optional[numeric] = None,
    marginal_ticks: bool = False,
    joint_kws: Optional[Dict[str, Any]] = None,
    marginal_kws: Optional[Dict[str, Any]] = None,
    hue: Optional[Union[str, List[numeric], ndarray]] = None,
    palette: Optional[
        Union[str, List[Union[numeric, str]], Dict[str, Any], Colormap]
    ] = None,
    hue_order: Optional[List[str]] = None,
    hue_norm: Optional[Union[Tuple[numeric], Normalize]] = None,
    **kwargs: Any,
) -> Tuple[JointGrid, pd.DataFrame]:
    if isinstance(hue, str):
        # Column with the same name
        # as the value of "hue" argument are not embedding target
        if data.columns.isin([hue]).sum():
            tmp_index_names = data.index.names
            data = data.reset_index().set_index(
                list(filter(None, tmp_index_names + [hue]))
            )
    X = data.values
    X_embedding: ndarray = embedding_transformer.fit_transform(X)
    df_embedding: pd.DataFrame = pd.DataFrame(
        data=X_embedding, index=data.index, columns=["x1", "x2"]
    )
    data_for_plot = df_embedding.reset_index(hue)
    g: JointGrid = sns.jointplot(
        data=data_for_plot,
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
    )
    g.ax_joint.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=math.ceil(math.sqrt(data_for_plot[hue].unique().size)),
    )
    return g, df_embedding
