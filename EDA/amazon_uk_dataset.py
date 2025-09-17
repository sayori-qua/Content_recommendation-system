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

path = kagglehub.dataset_download("asaniczka/amazon-uk-products-dataset-2023")
model = SentenceTransformer('all-mpnet-base-v2')

for filename in os.listdir(path):
    full_file_name = os.path.join(path, filename)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dataset_dir)

df_uk = pd.read_csv(fr"{dataset_dir}/amz_uk_processed_data.csv")

df_uk = df_uk.dropna()
df_uk['stars'] = pd.to_numeric(df_uk['stars'], errors='coerce')
df_uk['asin'] = df_uk['asin'].fillna('')
df_uk['title'] = df_uk['title'].fillna('')
df_uk['imgUrl'] = df_uk['imgUrl'].fillna('')
df_uk['price'] = df_uk['price'].fillna('')
df_uk['productURL'] = df_uk['productURL'].fillna('')
df_uk['reviews'] = df_uk['reviews'].fillna('')
df_uk['isBestSeller'] = df_uk['isBestSeller'].fillna('')
df_uk['boughtInLastMonth'] = df_uk['boughtInLastMonth'].fillna('')
df_uk['categoryName'] = df_uk['categoryName'].fillna('')
print(df_uk['categoryName'].unique())

target_categories = ['Lighting', 'Smart Speakers', 'Cameras', 'Torches', 'Coffee & Espresso Machines',
                     'Car & Motorbike', 'Smartwatches', 'Binoculars, Telescopes & Optics', 'Clocks', 'GPS, Finders & Accessories',
                     'Hi-Fi Receivers & Separates', 'Telephones, VoIP & Accessories']

target_lower = [cat.lower() for cat in target_categories]
mask = df_uk['categoryName'].str.lower().apply(lambda x: any(x.startswith(prefix) for prefix in target_lower))

df_uk = df_uk[mask].copy()
MAX_PER_CATEGORY = 60

def take_first_n(group):
    return group.head(MAX_PER_CATEGORY)

df_uk = df_uk.groupby('categoryName', group_keys=False).apply(take_first_n)
df_uk.reset_index(drop=True, inplace=True)

df_uk = df_uk.drop_duplicates()
category_counts = df_uk['categoryName'].value_counts()
df_uk = df_uk.reset_index(drop=True)
print(category_counts)


df_uk['combined_features'] = (
    df_uk['title'].astype(str) + ' ' +
    df_uk['categoryName'].astype(str) + ' ' +
    df_uk['price'].astype(str) + ' ' +
    df_uk['reviews'].astype(str)
)

print(df_uk['categoryName'].unique())

output_dir = 'graphics'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(18,13))

group1 = df_uk.groupby('categoryName')['price'].mean().sort_values(ascending=False)
top_10_categories = group1.head(10)
short_labels5 = [label.split('|')[-1].strip() for label in top_10_categories.index]
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(short_labels5))) #создаем массив цветов
bars = plt.bar(short_labels5, top_10_categories.values, color=colors, edgecolor='black')
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height ,
        f'{height:.1f}',
        ha='center', #выравнивание текста по горизонтали
        va='bottom',  #выравнивание текста по вертикали
        fontsize=20,
        fontweight='bold' #жирный шрифт
    )
plt.title('Mean price in top 10 categories', fontsize=30)
plt.xlabel('Category', fontsize=24)
plt.ylabel('Price', fontsize=24)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.ylim(10, 135)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(os.path.join(output_dir, 'mean_price.png'), dpi=200, bbox_inches='tight')
plt.show()

group2 = df_uk.groupby('categoryName')['reviews'].mean().sort_values(ascending=False)
top_10_categories_ = group2.head(10)
short_labels = [label.split('|')[-1].strip() for label in top_10_categories_.index]
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(short_labels)))
plt.figure(figsize=(16, 12))
bars = plt.bar(short_labels, top_10_categories_.values, color=colors, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.1f}',
        ha='center',
        va='bottom',
        fontsize=20,
        fontweight='bold'
    )
plt.title('Mean quantity reviews in top 10 category', fontsize=30)
plt.xlabel('Category', fontsize=24)
plt.ylabel('Quantity reviews', fontsize=24)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(os.path.join(output_dir, 'Mean_quantity.png'), dpi=200)
plt.show()

