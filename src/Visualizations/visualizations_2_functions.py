import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def plot_avg_intra_genre_cosine_by_decade(
    df,
    vector_col='vector',
    genre_col='Bucket',
    year_col='Year_Extracted',
    decade_col=None,
    y_range=None
):
    """
    Compute average intra-genre cosine‐similarity by decade and plot it.
    """
    # work on a copy
    df = df.copy()

    # 1) ensure we have a “decade” column
    if decade_col is None:
        df['decade'] = (df[year_col] // 10) * 10
        decade_col = 'decade'

    # 2) compute centroids for each (genre, decade)
    centroids = (
        df
        .groupby([genre_col, decade_col])[vector_col]
        .apply(lambda vs: np.mean(np.stack(vs), axis=0))
        .reset_index()
        .rename(columns={vector_col: 'centroid'})
    )

    # 3) merge back and compute cosine‐similarity to its centroid
    df = df.merge(centroids, on=[genre_col, decade_col])
    df['cos_sim'] = df.apply(
        lambda row: cosine_similarity(
            row[vector_col].reshape(1, -1),
            row['centroid'].reshape(1, -1)
        )[0,0],
        axis=1
    )

    # 4) average per (genre, decade)
    summary = (
        df
        .groupby([genre_col, decade_col])['cos_sim']
        .mean()
        .reset_index()
    )

    # 5) plot
    fig = px.line(
        summary,
        x=decade_col,
        y='cos_sim',
        color=genre_col,
        markers=True,
        title="Average Intra‐Genre Cosine Similarity by Decade",
        labels={
            decade_col: "Decade",
            'cos_sim': "Avg. Cosine Similarity",
            genre_col: "Genre"
        }
    )
    fig.update_layout(legend_title_text='Genre')
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig



def plot_cos_sim_bubble_chart(
    df,
    vector_col='vector',
    genre_col='Bucket',
    decade_col='Decade',
    sim_col='cos_sim',
    y_range=(0.55, 0.71)
):
    """
    Plots a bubble chart of average cosine similarity by genre and decade.
    
    Parameters:
    - df : pandas.DataFrame
        DataFrame with cosine similarity values.
    - genre_col : str
        Column name for genre (default: 'Bucket').
    - decade_col : str
        Column name for decade (default: 'decade').
    - sim_col : str
        Column name for cosine similarity (default: 'cos_sim').
    - y_range : tuple or None
        Optional y-axis range as (min, max).
    """
    # work on a copy
    df = df.copy()

    # 1) ensure we have a “decade” column
    if decade_col is None:
        df['decade'] = (df['Year_Extracted'] // 10) * 10
        decade_col = 'decade'

    # 2) compute centroids for each (genre, decade)
    centroids = (
        df
        .groupby([genre_col, decade_col])[vector_col]
        .apply(lambda vs: np.mean(np.stack(vs), axis=0))
        .reset_index()
        .rename(columns={vector_col: 'centroid'})
    )

    # 3) merge back and compute cosine‐similarity to its centroid
    df = df.merge(centroids, on=[genre_col, decade_col])
    df['cos_sim'] = df.apply(
        lambda row: cosine_similarity(
            row[vector_col].reshape(1, -1),
            row['centroid'].reshape(1, -1)
        )[0,0],
        axis=1
    )

    stats = (
        df
        .groupby([genre_col, decade_col])[sim_col]
        .agg(count='size', mean_sim='mean')
        .reset_index()
    )

    fig = px.scatter(
        stats,
        x=decade_col,
        y='mean_sim',
        color=genre_col,
        size='count',
        hover_data=['count'],
        title="Mean Cos‐Sim by Genre & Decade (bubble ∝ # songs)"
    )

    if y_range:
        fig.update_yaxes(range=y_range)

    return fig





def plot_decade_bar_chart(decade_similarities, y_range=(0, 0.5)):
    """
    Plots a bar chart of average cosine similarity by decade using Plotly.

    Parameters:
    - decade_similarities: dict
        Dictionary with decade as key and average cosine similarity as value.
    - y_range: tuple
        Tuple to set y-axis range (default is (0, 0.5)).

    Returns:
    - fig: plotly.graph_objects.Figure
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame({
        'Decade': list(decade_similarities.keys()),
        'Average Cosine Similarity': list(decade_similarities.values())
    }).sort_values(by='Decade')

    # Plot with Plotly
    fig = px.bar(
        df,
        x='Decade',
        y='Average Cosine Similarity',
        title='Lyrical Similarity by Decade',
        labels={'Decade': 'Decade', 'Average Cosine Similarity': 'Avg. Cosine Similarity'},
        text='Average Cosine Similarity'
    )

    fig.update_traces(marker_color='skyblue', texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        yaxis_range=y_range,
        xaxis_tickangle=-45,
        yaxis_title='Average Cosine Similarity',
        xaxis_title='Decade',
        showlegend=False,
        bargap=0.2,
        margin=dict(l=40, r=40, t=60, b=60)
    )
    return fig


