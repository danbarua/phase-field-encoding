# %%
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def generate_phase_plot(Z, labels, title="Phase Space Trajectory", use_tsne=False):
    """
    Z: np.ndarray of shape [T, N, D] where
       T = timesteps or layers,
       N = number of samples,
       D = phase-feature dimension (e.g. 2*neurons)

    labels: np.ndarray of shape [N] — class labels per sample
    """
    T, N, D = Z.shape
    all_points = []

    for t in range(T):
        if use_tsne:
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
        else:
            reducer = PCA(n_components=2)

        points_2d = reducer.fit_transform(Z[t])

        df = pd.DataFrame(
            {
                "x": points_2d[:, 0],
                "y": points_2d[:, 1],
                "class": labels.astype(str),
                "timestep": t,
            }
        )
        all_points.append(df)

    df_all = pd.concat(all_points)

    fig = px.scatter(
        df_all,
        x="x",
        y="y",
        color="class",
        animation_frame="timestep",
        title=title,
        labels={"class": "Digit Class"},
        opacity=0.7,
        width=900,
        height=700,
    )

    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    fig.update_layout(template="plotly_dark", transition_duration=300)
    fig.show()
