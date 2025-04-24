import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

def plot_genre_similarity_by_decade_grid(df, vector_col='vector', genre_col='Bucket', decade_col='Decade'):
    """
    Creates a faceted bar plot showing average genre similarity per decade.

    Parameters:
    - df: DataFrame containing vectors, genres, and decades
    - vector_col: name of the column with vector data
    - genre_col: name of the column with genre (bucket) info
    - decade_col: name of the column with decade info

    Returns:
    - Plotly figure
    """
    records = []

    for (decade, genre), group in df.groupby([decade_col, genre_col]):
        if len(group) > 1:
            vectors = np.stack(group[vector_col])
            sims = cosine_similarity(vectors)
            mask = ~np.eye(len(sims), dtype=bool)
            avg_sim = sims[mask].mean()
            records.append({
                "Decade": decade,
                "Genre": genre,
                "Avg_Similarity": avg_sim
            })

    plot_df = pd.DataFrame(records)

    fig = px.bar(
        plot_df,
        x="Genre",
        y="Avg_Similarity",
        facet_col="Decade",
        facet_col_wrap=2,
        title="Average Lyrical Similarity by Genre Across Decades",
        labels={"Avg_Similarity": "Average Cosine Similarity"},
        color="Genre",
        category_orders={"Decade": sorted(plot_df["Decade"].unique())}
    )

    fig.update_layout(
        height=400 + 200 * (len(plot_df["Decade"].unique()) // 2),
        showlegend=False
    )

    fig.update_yaxes(range=[0, 0.5])
    fig.for_each_xaxis(lambda axis: axis.update(tickangle=45))
    return fig
