import vis_functions as vis
import pandas as pd

df = pd.read_csv("C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/data/Improved_Bucket_Accuracy_balanced_60s_up.csv")

decades_list = vis.get_decades(df)
vis.decades_scatter(decades_list, 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/Visualizations')
vis.decades_radial(decades_list, 'C:/Users/Daniel Day/Downloads/School/Data-399/data_capstone_399/Visualizations')
vis.plotly_scatter_explicit(df) 