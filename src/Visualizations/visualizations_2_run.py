import pandas as pd
import numpy as np
import visualizations_2_functions as vis2  # the file where your function is defined

# 1. Load your dataset
df = pd.read_csv("Improved_Bucket_Accuracy_all_songs_raw_with_buckets.csv")

# Convert the 'vector' column from string to actual NumPy arrays
import ast
df['vector'] = df['vector'].apply(ast.literal_eval).apply(lambda x: np.array(x))

# 2. Generate and show the line plot of average intra-genre cosine similarity
fig1 = vis2.plot_avg_intra_genre_cosine_by_decade(df)
fig1.show()

# 3. Generate and show the bubble chart for genre/decade similarity and song count
fig2 = vis2.plot_cos_sim_bubble_chart(df)
fig2.show()


from sklearn.metrics.pairwise import cosine_similarity

# 2. Compute average cosine similarity per decade
decade_similarities = {}
for decade, group in df.groupby("Decade"):
    vectors = np.stack(group["vector"])
    sims = cosine_similarity(vectors)

    # Exclude self-similarity (the diagonal)
    mask = ~np.eye(len(sims), dtype=bool)
    avg_sim = sims[mask].mean()

    decade_similarities[decade] = avg_sim

fig3 = vis2.plot_decade_bar_chart(decade_similarities)
fig3.show()

