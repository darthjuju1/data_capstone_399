import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import math



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
