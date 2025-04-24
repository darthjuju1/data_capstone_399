import pandas as pd
import numpy as np
import visualizations_3_functions as vis3 

df = pd.read_csv(r"data\Improved_Bucket_Accuracy_all_songs_raw_with_buckets.csv")

import ast
df['vector'] = df['vector'].apply(ast.literal_eval).apply(lambda x: np.array(x))


fig = vis3.plot_genre_similarity_by_decade_grid(df)
fig.show()
