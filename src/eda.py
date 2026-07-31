import pandas as pd

df = pd.read_csv("data/raw/student_placement_data.csv")
print(df.columns.tolist())