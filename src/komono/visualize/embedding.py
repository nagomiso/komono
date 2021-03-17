from typing import Any, Optional, Union, List
from numbers import Number

import pandas as pd
from matplotlib.colors import Colormap, Normalize
from numpy import ndarray
import seaborn as sns
from seaborn import JointGrid


def jointplot(
    data: pd.DataFrame,
    embedding_transformer: Any,
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
    hue_norm: Optional[tuple, Normalize] = None,
    **kwargs
) -> JointGrid:
    embedding: ndarray = embedding_transformer.fit_transform(data.values)
    df_embedding: pd.DataFrame = pd.DataFrame(
        data=embedding, index=data.index, columns=["x1", "x2"]
    )
    return sns.jointplot(
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
    )
