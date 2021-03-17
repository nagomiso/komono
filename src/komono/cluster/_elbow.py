from typing import List, Union

import seaborn as sns
from numpy import ndarray
from sklearn.cluster import KMeans, MiniBatchKMeans


class Elbow(object):
    def __init__(
        self,
        kmeans: Union[KMeans, MiniBatchKMeans],
        kmin: int,
        kmax: int,
        draw: bool = False,
    ) -> None:
        self._kmeans = kmeans
        self._kmin = kmin
        self._kmax = kmax
        self._draw = draw
        self.optimal_n_clusters = -1

    def fit(self, X: Union[list, ndarray]) -> "Elbow":
        n_clusters: List[int] = []
        inertias: List[float] = []
        for k in range(self._kmin, self._kmax):
            self._kmeans.set_params(n_clusters=k)
            self._kmeans.fit(X)
            n_clusters.append(k)
            inertias.append(self._kmeans.inertia_)
        self.optimal_n_clusters = self._find_optimal_n_clusters(
            n_clusters=n_clusters, inertias=inertias
        )
        if self._draw:
            self._draw_elbow_chart(n_clusters=n_clusters, inertias=inertias)
        return self

    @staticmethod
    def _find_optimal_n_clusters(n_clusters: List[int], inertias: List[float]) -> int:
        # The implementation of this function is based on pyclustering.
        # ref: https://github.com/annoviko/pyclustering/blob/0.10.1.2/pyclustering/cluster/elbow.py#L186
        max_score: float = -1.0
        optimal_n_clusters: int = -1

        x_kmin, y_kmin = 0.0, inertias[0]
        x_kmax, y_kmax = len(n_clusters), inertias[-1]
        const = x_kmin * y_kmax - x_kmax * y_kmin

        # Calculating distance from each point (x_k, y_k) to line segment from
        # kmin-point (x_kmin, y_kmax) to kmax-point (x_kmax, y_kmin)
        # and the point where is the longest distance is considered optimal point.
        #
        # The formula for distance is bellow.
        #
        #     |(y_kmin - y_kmax) * x_k + (x_kmax - x_kmin) * y_k|
        #     |     + (x_kmin * y_kmax - x_kmax * y_kmin)       |
        #   -------------------------------------------------------
        #    sqrt((x_kmax - x_kmin) ** 2 + (y_kmax - y_kmin) ** 2)
        #
        # But, the denominator is constant in the case of search the point
        # where is the longest distance, so we can ignore the denominator.
        for x_k, y_k in zip(n_clusters, inertias):
            score = abs((y_kmin - y_kmax) * x_k + (x_kmax - x_kmin) * y_k + const)
            if max_score < score:
                max_score = score
                optimal_n_clusters = x_k
        return optimal_n_clusters

    def _draw_elbow_chart(self, n_clusters: List[int], inertias: List[float]) -> None:
        ax = sns.lineplot(
            x=n_clusters,
            y=inertias,
            markers=["o"],
            style=0,
            legend=False,
        )
        ax.set_xlabel("N Clusters")
        ax.set_ylabel("Inertias")
        ax.set_xticks(n_clusters)
        ax.vline(
            x=self.optimal_n_clusters,
            ymin=min(inertias),
            ymax=max(inertias),
            colors="black",
            linestyle="--",
            linewidths=2,
        )
