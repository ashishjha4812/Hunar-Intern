import pandas as pd
import numpy as np

#read
df = pd.read_csv(r"E:\Users\oma\Downloads\food_coded.csv")

# quick look
df.head()

print("Shape before cleaning:", df.shape)
print("\nInfo:")
df.info()

print("\nMissing values per column:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# separate column types
num_cols  = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols  = df.select_dtypes(exclude=['int64', 'float64']).columns

# ---- Numerical columns ----
# Example: use mean (you can switch to median if needed)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# If you want median instead, use:
# df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ---- Categorical columns ----
for col in cat_cols:
    mode_value = df[col].mode()
    if not mode_value.empty:
        df[col].fillna(mode_value[0], inplace=True)


print("Missing values after filling:")
print(df.isna().sum())

print("Duplicate rows before:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate rows after:", df.duplicated().sum())


print("Columns before removing duplicates:", len(df.columns))

# remove columns with exactly same values
df = df.loc[:, ~df.T.duplicated()]

print("Columns after removing duplicates:", len(df.columns))

print("Final shape:", df.shape)
print("\nFinal info:")
df.info()


df.to_csv("hunar_task1_cleaned.csv", index=False)
print("Cleaned file saved as 'hunar_task1_cleaned.csv'")



