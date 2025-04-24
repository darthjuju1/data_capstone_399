import vis_functions as vis
import pandas as pd

df = pd.read_csv("C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/Improved_Bucket_Accuracy_balanced_60s_up.csv")
filepath = "C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/src/Visualizations"

decades_list = vis.get_decades(df)
vis.decades_scatter(decades_list, filepath=filepath)
vis.decades_radial(decades_list, 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/Visualizations')
vis.plotly_scatter_explicit(df) 
vis.plot_avg_intra_genre_cosine_by_decade(df, filepath=filepath)
vis.plot_cos_sim_bubble_chart(df, filepath=filepath)
vis.plot_decade_bar_chart(df, filepath=filepath)
vis.decades_genre_similarity(decades_list, filepath=filepath)