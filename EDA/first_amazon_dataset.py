import kagglehub
import os
import torch
import shutil
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

dataset_dir = r"/dataset"
os.makedirs(dataset_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path = kagglehub.dataset_download("karkavelrajaj/amazon-sales-dataset")
model = SentenceTransformer('all-mpnet-base-v2')

for filename in os.listdir(path):
    full_file_name = os.path.join(path, filename)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dataset_dir)

df = pd.read_csv(fr"{dataset_dir}/amazon.csv")
df = df.dropna()
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['user_id'] = df['user_id'].str.split(',')
df['product_name'] = df['product_name'].fillna('')
df['review_content'] = df['review_content'].fillna('')
df['about_product'] = df['about_product'].fillna('')
df['category'] = df['category'].fillna('')
df['actual_price'] = df['actual_price'].fillna('')
df['discount_percentage'] = df['discount_percentage'].fillna('')
df['about_product'] = df['about_product'].str.replace('|', '', regex=False)
df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)

df_exploded = df.explode('user_id')
df_exploded = df_exploded.drop_duplicates(subset=['product_name', 'about_product'])
df_exploded = df_exploded.reset_index(drop=True)
df_exploded['product_id'] = df_exploded.index
category_counts = df_exploded['category'].value_counts()
print(category_counts)

#data для contenta
df_exploded['combined_features'] = (
    df_exploded['product_name'].astype(str) + ' ' +
    df_exploded['review_content'].astype(str) + ' ' +
    df_exploded['about_product'].astype(str) + ' ' +
    df_exploded['category'].astype(str) + ' ' +
    df_exploded['actual_price'].astype(str) + ' ' +
    df_exploded['discount_percentage'].astype(str)
)

#анализ
output_dir = r'/EDA\graphics'
os.makedirs(output_dir, exist_ok=True)
plt.figure(figsize=(20, 10))
group1 = df_exploded.groupby('category')['rating'].mean().sort_values(ascending=False)
top_10_categories = group1.head(10)
short_labels5 = [label.split('|')[-1].strip() for label in top_10_categories.index]
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(short_labels5)))
bars = plt.bar(short_labels5, top_10_categories.values, color=colors, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.05,
        f'{height:.1f}',
        ha='center',
        va='bottom',
        fontsize=20,
        fontweight='bold'
    )
plt.title('Mean rating in top 10 categories', fontsize=30)
plt.xlabel('Category', fontsize=24)
plt.ylabel('Rating', fontsize=24)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.ylim(4, 4.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(os.path.join(output_dir, 'mean_rating.png'), dpi=200, bbox_inches='tight')
plt.show()

group2 = df_exploded.groupby('category')['discount_percentage'].mean().sort_values(ascending=False)
top_5_categories = group2.head(5)
short_labels = [label.split('|')[-1].strip() for label in top_5_categories.index]
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(short_labels)))
plt.figure(figsize=(12, 10))
bars = plt.bar(short_labels, top_5_categories.values, color=colors, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f'{height:.1f}%',
        ha='center',
        va='bottom',
        fontsize=20,
        fontweight='bold'
    )
plt.title('Discount percentage in top 5 categories', fontsize=30)
plt.xlabel('Category', fontsize=24)
plt.ylabel('Discount percentage (%)', fontsize=24)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.ylim(86, 91)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(os.path.join(output_dir, 'Discount_percentage_gradient.png'), dpi=200)
plt.show()

