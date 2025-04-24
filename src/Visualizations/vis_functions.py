import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity



def get_decades(df):
    """
    Description
    Splits the DataFrame into separate DataFrames for each decade.
    -----------
    Paramaters
    df : pandas DataFrame of Lyrics Data
    ----------
    Returns
    decades_list : list of DataFrames
    """
    df_1960s = df[df['Decade'] == '1960s']
    df_1970s = df[df['Decade'] == '1970s']
    df_1980s = df[df['Decade'] == '1980s']
    df_1990s = df[df['Decade'] == '1990s']
    df_2000s = df[df['Decade'] == '2000s']
    df_2010s = df[df['Decade'] == '2010s']
    df_2020s = df[df['Decade'] == '2020s']
    
    decades_list = [df_1960s, df_1970s, df_1980s, df_1990s, df_2000s, df_2010s, df_2020s]

    return decades_list

def plotly_scatter(df, year, filepath):
    """
    Description
    Takes in a DataFrame consisting of PCA data of lyrical similarity across genres and decades, and produces an interactive scatterplot.
    Calculates Centroids for each genre bucket.
    Assigns colors to each genre bucket then plots the data points and centroids.
    Saves the plotly visualization to an HTML
    -----------
    Paramaters
    df : pandas DataFrame
    year : string representing the decade of music.
    ----------
    Returns
    fig : plotly Figure object
    ----------
    """
    # Calculate centroids for each cluster/bucket
    centroids = df.groupby('Bucket')[['PC1', 'PC2']].mean().reset_index()

    
    # Store bucket colors
    bucket_colors = {}
    for i, bucket in enumerate(df['Bucket'].unique()):
        bucket_colors[bucket] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
    
    # Create figure and add regular data points
    fig = go.Figure()
    for bucket in df['Bucket'].unique():
        bucket_data = df[df['Bucket'] == bucket]
        fig.add_trace(
            go.Scatter(
                x=bucket_data['PC1'],
                y=bucket_data['PC2'],
                mode='markers',
                name=bucket,
                hovertext=bucket_data['Track Name'],
                marker=dict(
                    size=8,
                    color=bucket_colors[bucket],
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                legendgroup=bucket,
            )
        )
    
    # Add centroids as separate traces on top
    for bucket in centroids['Bucket'].unique():
        centroid = centroids[centroids['Bucket'] == bucket]
        fig.add_trace(
            go.Scatter(
                x=centroid['PC1'],
                y=centroid['PC2'],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=20,
                    color=bucket_colors[bucket],
                    line=dict(width=2, color='black'),
                    opacity=1.0
                ),
                name=f'Centroid: {bucket}',
                legendgroup=bucket,
                showlegend=True
            )
        )

    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        xaxis=dict(
            title='PC1',
            gridcolor='lightgray',
            zerolinecolor='lightgray',
        ),
        yaxis=dict(
            title='PC2',
            gridcolor='lightgray',
            zerolinecolor='lightgray',
        ),
        plot_bgcolor='white',
        legend_title_text='Genre Bucket',
        title=f"{year}'s Songs by Genre and Similarity score"
    )

    # save the plot as an HTML file for embedding
    fig.write_html(f'{filepath}/{year}_pca_scatterplot.html')

def decades_scatter(decades_list, filepath):
    """
    Description
    Establish list of decades to iterate through and call the plotly_scatter function for each decade.
    Create a scatter plot for each decade in the provided list.

    Parameters
    -----------
    decades_list : list of DataFrames
        List containing DataFrames for each decade
    """
    for i, decade_df in enumerate(decades_list):
        decade_name = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'][i]
        plotly_scatter(decade_df, decade_name, filepath)
        return filepath

def plotly_combined_line_graph(df):
    """
    Description
    Create a line graph showing the evolution of genres across decades based on 
    the combined absolute values of PC1 and PC2.
    ----------
    
    Parameters:
    df : pandas DataFrame
    ----------
    
    Returns:
    fig : plotly Figure object
    ----------
    """
    
    # Create a new column with the combined absolute values
    df['Combined_PC'] = np.abs(df['PC1']) + np.abs(df['PC2'])
    
    # Calculate statistics for each bucket and decade
    stats = df.groupby(['Bucket', 'Decade']).agg({
        'Combined_PC': ['mean', 'std']
    }).reset_index()
    
    # Flatten the multi-level columns
    stats.columns = ['Bucket', 'Decade', 'Combined_PC_mean', 'Combined_PC_std']
    
    # Create figure
    fig = go.Figure()
    
    # Get unique buckets in alphabetical order
    buckets = sorted(df['Bucket'].unique())
    
    # Define a color palette
    colors = [
        'red', 'orange', 'teal', 'green', 'indigo',
        'blue', 'violet', 'pink', 'brown'
    ]
    
    # Plot for each bucket
    for i, bucket in enumerate(buckets):
        color = colors[i % len(colors)]
        bucket_stats = stats[stats['Bucket'] == bucket].sort_values('Decade')
        
        # Main line
        fig.add_trace(
            go.Scatter(
                x=bucket_stats['Decade'],
                y=bucket_stats['Combined_PC_mean'],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=10, symbol='circle'),
                name=bucket,
                legendgroup=bucket,
                showlegend=True,
                hovertemplate=f"{bucket}<br>Decade: %{{x}}<br>Combined PC: %{{y:.2f}}<extra></extra>"
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Genre Evolution Across Decades (Combined PC Metric)',
        height=600,
        width=1000,
        legend_title_text='Genre',
        hovermode='closest',
        xaxis=dict(
            title='Decade',
            type='category'
        ),
        yaxis=dict(title='Average Distance away from Combined PC'),
        plot_bgcolor='white'
    )
    
    # Update axis line styles
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray')
    
    return fig

def create_genre_radial_plot(df, Decade, filepath):
    """
    Description
    Create a radial plot for each genre showing the distribution of songs
    around their genre centroids.
    ----------
    
    Parameters:
    df : pandas DataFrame containing columns: 'Bucket', 'PC1', 'PC2', 'Track Name', 'Artist Name' and 'Decade'
    Decade : str, optional
        Filter data for a specific decade if provided (e.g., '1960s')

    Returns:
    --------
    fig : The created radial plot visualization as a plotly Figure
    """
    # Filter by decade
    df = df[df['Decade'] == Decade].copy()
    
    # Calculate centroids for each bucket
    centroids = df.groupby('Bucket')[['PC1', 'PC2']].mean().reset_index()
    all_buckets = sorted(df['Bucket'].unique())
    
    # Create subplot grid
    fig = make_subplots(
        rows=3, 
        cols=3,
        subplot_titles=[f"Genre: {bucket}" for bucket in all_buckets],
        specs=[[{"type": "polar"} for _ in range(3)] for _ in range(3)]
    )
    
    # Plot each genre
    for i, bucket in enumerate(all_buckets):
        # Calculate row and column for this subplot
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Get data for this genre
        genre_df = df[df['Bucket'] == bucket].copy()
        
        # Get centroid for this genre
        centroid = centroids[centroids['Bucket'] == bucket].iloc[0]
        centroid_pc1 = centroid['PC1']
        centroid_pc2 = centroid['PC2']
        
        # Calculate polar coordinates relative to centroid
        # (Convert from Cartesian to polar coordinates)
        genre_df['dx'] = genre_df['PC1'] - centroid_pc1
        genre_df['dy'] = genre_df['PC2'] - centroid_pc2
        genre_df['distance'] = np.sqrt(genre_df['dx']**2 + genre_df['dy']**2)
        genre_df['theta'] = np.arctan2(genre_df['dy'], genre_df['dx']) * 180 / np.pi
        
        # Convert theta to 0-360 range
        genre_df['theta'] = genre_df['theta'] % 360
        
        # Add trace for songs
        hover_text = []
        for _, song in genre_df.iterrows():
            hover_info = []
            hover_info.append(f"Track: {song['Track Name']}")
            hover_info.append(f"Artist: {song['Artist Name']}")
            hover_info.append(f"Year: {song['Year_Extracted']}")
            hover_info.append(f"Distance: {song['distance']:.2f}")
            hover_text.append("<br>".join(hover_info))
        
        fig.add_trace(
            go.Scatterpolar(
                r=genre_df['distance'],
                theta=genre_df['theta'],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7,
                ),
                name=bucket,
                text=hover_text,
                hoverinfo='text',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add a reference circle at the median distance
        median_distance = genre_df['distance'].median()
        theta_values = np.linspace(0, 360, 100)
        r_values = [median_distance] * 100
        
        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                mode='lines',
                line=dict(
                    color='rgba(0,0,0,0.3)',
                    dash='dash'
                ),
                name='Median Distance',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=f"Song Distribution Around Genre Centroids{' for ' + str(Decade) if Decade else ''}",
        height=300 * 3,
        width=300 * 3,
        margin=dict(l=30, r=30, t=80, b=30),  # Increased top margin
        template="plotly_white"
    )
    
    # Update subplot titles to be positioned higher
    for i in fig['layout']['annotations']:
        i['y'] = i['y']
    
    # Update polar axes to have consistent scale across subplots
    
    fig.update_polars(
        radialaxis=dict(
            visible=True,
            range=[0, 0.5]  # Set max range to 0.5 as requested
        ),
        angularaxis=dict(
            visible=True,
            rotation=90,  # Start at top (North)
            direction="clockwise"
        )
    )

    fig.write_html(f'{filepath}/{Decade}_radial.html')
    return fig

def decades_radial(decades_list, filepath):
    """
    Create a genre radial plot for each decade in the provided list.

    Parameters:
    -----------
    decades_list : list of DataFrames
    List containing DataFrames for each decade (e.g., [df_1960s, df_1970s, ...])
    """
    for i, decade_df in enumerate(decades_list):
        decade_name = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'][i]
        create_genre_radial_plot(decade_df, decade_name, filepath)
        return filepath

def plotly_scatter_explicit(df, filepath):
    """
    Create a scatterplot visualization where points are colored by Explicit status
    instead of genre buckets.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing columns: 'PC1', 'PC2', 'Explicit', 'Track Name'
        
    Returns:
    --------
    fig : plotly Figure object
        The created scatterplot visualization
    """
    # Create figure
    fig = go.Figure()
    
    # Define colors for explicit vs non-explicit
    colors = {
        True: "#e74c3c",   # Red for explicit
        False: "#3498db"    # Blue for non-explicit
    }
    
    # Create traces for each explicit status
    for explicit_status in [False, True]:  # Order matters for legend (False first, then True)
        status_label = "Explicit" if explicit_status else "Clean"
        status_data = df[df['Explicit'] == explicit_status]
        
        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=status_data['PC1'],
                y=status_data['PC2'],
                mode='markers',
                name=status_label,
                hovertext=[
                    f"Track: {row['Track Name']}<br>"
                    f"Artist: {row['Artist Name']}<br>"
                    f"Year: {row['Year_Extracted']}<br>"
                    f"Genre: {row['Bucket']}"
                    for _, row in status_data.iterrows()
                ],
                marker=dict(
                    size=8,
                    color=colors[explicit_status],
                    opacity=0.7,
                    line=dict(width=1, color='DarkSlateGrey')
                )
            )
        )
    
    # Calculate centroids for each explicit status group
    centroids = df.groupby('Explicit')[['PC1', 'PC2']].mean().reset_index()
    
    # Add centroids
    for _, centroid in centroids.iterrows():
        explicit_status = centroid['Explicit']
        status_label = "Explicit" if explicit_status else "Clean"
        
        fig.add_trace(
            go.Scatter(
                x=[centroid['PC1']],
                y=[centroid['PC2']],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=20,
                    color=colors[explicit_status],
                    line=dict(width=2, color='black'),
                    opacity=1.0
                ),
                name=f'Centroid: {status_label}',
                showlegend=True
            )
        )
        
        # Add text labels to centroids
        fig.add_trace(
            go.Scatter(
                x=[centroid['PC1']],
                y=[centroid['PC2']],
                mode='text',
                text=[f'Centroid: {status_label}'],
                textposition='top center',
                textfont=dict(size=10, color='black'),
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        xaxis=dict(
            title='PC1',
            gridcolor='lightgray',
            zerolinecolor='lightgray',
        ),
        yaxis=dict(
            title='PC2',
            gridcolor='lightgray',
            zerolinecolor='lightgray',
        ),
        plot_bgcolor='white',
        legend_title_text='Explicit Content',
        title=f"Songs by Explicit Content and Similarity Score"
    )
    fig.write_html(f'{filepath}/explicit_scatterplot.html')
    return fig

def plot_avg_intra_genre_cosine_by_decade(
    df,
    filepath,
    vector_col='vector',
    genre_col='Bucket',
    year_col='Year_Extracted',
    decade_col=None,
    y_range=None,
):
    """
    Compute average intra-genre cosine‐similarity by decade and plot it.
    """
    import ast
    df['vector'] = df['vector'].apply(ast.literal_eval).apply(lambda x: np.array(x))
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
    fig.write_html(f'{filepath}/avg_intra_genre_cosine_by_decade.html')
    return fig

def plot_cos_sim_bubble_chart(
    df,
    filepath,
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
    import ast
    df['vector'] = df['vector'].apply(ast.literal_eval).apply(lambda x: np.array(x))

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

    fig.write_html(f'{filepath}/cos_sim_bubble_chart.html')

    return fig

def plot_decade_bar_chart(df, filepath, y_range=(0, 0.5)):
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

    # Convert the 'vector' column from string to actual NumPy arrays
    import ast
    df['vector'] = df['vector'].apply(ast.literal_eval).apply(lambda x: np.array(x))

    # 2. Compute average cosine similarity per decade
    decade_similarities = {}
    for decade, group in df.groupby("Decade"):
        vectors = np.stack(group["vector"])
        sims = cosine_similarity(vectors)

    # Exclude self-similarity (the diagonal)
    mask = ~np.eye(len(sims), dtype=bool)
    avg_sim = sims[mask].mean()

    decade_similarities[decade] = avg_sim

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

    fig.write_html(f'{filepath}/decade_bar_chart.html')

    return fig

def plot_genre_similarity_single_decade(df, filepath, decade, vector_col='vector', genre_col='Bucket'):
    """
    Creates a bar plot showing average genre similarity for a single decade/year.

    Parameters:
    - df: DataFrame containing vectors and genres for a single decade
    - vector_col: name of the column with vector data
    - genre_col: name of the column with genre (bucket) info

    Returns:
    - Plotly figure
    """
    import ast
    df['vector'] = df['vector'].apply(ast.literal_eval).apply(lambda x: np.array(x))

    records = []
    
    # Get the decade name for the title (assuming all rows have the same decade)
    decade = df['Decade'].iloc[0] if 'Decade' in df.columns else "Unknown"
    
    # For each genre, calculate the average pairwise similarity
    for genre, group in df.groupby(genre_col):
        # Skip if there's only one track (need at least 2 for similarity)
        if len(group) > 1:
            try:
                # Stack the vectors for this genre
                vectors = np.stack(group[vector_col].values)
                
                # Calculate pairwise similarities
                sims = cosine_similarity(vectors)
                
                # Create a mask to exclude self-similarities (diagonal)
                mask = ~np.eye(len(sims), dtype=bool)
                
                # Calculate the average similarity
                avg_sim = sims[mask].mean()
                
                # Create a record for this genre with its sample size
                records.append({
                    "Genre": genre,
                    "Avg_Similarity": avg_sim,
                    "Count": len(group)
                })
            except Exception as e:
                print(f"Error processing genre {genre}: {str(e)}")
                print(f"Sample vector type: {type(group[vector_col].iloc[0])}")
                if isinstance(group[vector_col].iloc[0], str):
                    print("Vector is stored as string - needs conversion to array")
                continue
    
    # Convert records to DataFrame
    plot_df = pd.DataFrame(records)
    
    # Sort by similarity for better visualization
    plot_df = plot_df.sort_values("Avg_Similarity", ascending=False)
    
    # Create the bar chart
    fig = px.bar(
        plot_df,
        x="Genre",
        y="Avg_Similarity",
        color="Genre",
        title=f"Average Lyrical Similarity by Genre for {decade}",
        labels={"Avg_Similarity": "Average Cosine Similarity", "Genre": "Genre"},
        hover_data=["Count"]  # Show count in hover data
    )
    
    # Add count labels above bars
    fig.add_trace(
        go.Scatter(
            x=plot_df["Genre"],
            y=plot_df["Avg_Similarity"] + 0.02,  # Slightly above the bars
            mode="text",
            text=plot_df["Count"].apply(lambda x: f"n={x}"),
            textposition="top center",
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=800,
        showlegend=False,
        yaxis_range=[0, 0.5],  # Consistent y-axis range
        xaxis_tickangle=45,  # Angled labels for better readability
        plot_bgcolor="white"
    )

    fig.write_html(f'{filepath}/{decade}_genre_similarity.html')
    
    return fig

def decades_genre_similarity(decades_list, filepath):
    """
    Description
    Establish list of decades to iterate through and call the plotly_scatter function for each decade.
    Create a scatter plot for each decade in the provided list.

    Parameters
    -----------
    decades_list : list of DataFrames
        List containing DataFrames for each decade
    """
    for i, decade_df in enumerate(decades_list):
        decade_name = ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'][i]
        plot_genre_similarity_single_decade(decade_df, filepath, decade_name)
        return filepath