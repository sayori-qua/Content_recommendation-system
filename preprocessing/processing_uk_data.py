import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
from sqlalchemy import create_engine

device = "cuda" if torch.cuda.is_available() else "cpu"

engine = create_engine('')
model = SentenceTransformer('all-mpnet-base-v2')

def url_points_to_image(url): #проверяем картинки на наличие
    try:
        response = requests.head(url, timeout=3)
        content_type = response.headers.get('Content-Type', '')
        return response.status_code == 200 and 'image' in content_type
    except Exception as e:
        return False

df_uk = pd.read_sql('SELECT * FROM amazon_products', engine)
print(df_uk.columns)

df_uk = df_uk.dropna()
df_uk['stars'] = pd.to_numeric(df_uk['stars'], errors='coerce')
df_uk['asin'] = df_uk['asin'].fillna('')
df_uk['title'] = df_uk['title'].fillna('')
df_uk['imgurl'] = df_uk['imgurl'].fillna('')
df_uk['price'] = df_uk['price'].fillna('')
df_uk['producturl'] = df_uk['producturl'].fillna('')
df_uk['reviews'] = df_uk['reviews'].fillna('')
df_uk['isbestseller'] = df_uk['isbestseller'].fillna('')
df_uk['boughtinlastmonth'] = df_uk['boughtinlastmonth'].fillna('')
df_uk['categoryname'] = df_uk['categoryname'].fillna('')
print(df_uk['categoryname'].unique())

target_categories = ['Lighting', 'Smart Speakers', 'Cameras', 'Torches', 'Coffee & Espresso Machines',
                     'Car & Motorbike', 'Smartwatches', 'Binoculars, Telescopes & Optics', 'Clocks', 'GPS, Finders & Accessories',
                     'Hi-Fi Receivers & Separates', 'Telephones, VoIP & Accessories']
target_lower = [cat.lower() for cat in target_categories]
mask = df_uk['categoryname'].str.lower().apply(lambda x: any(x.startswith(prefix) for prefix in target_lower))

df_uk = df_uk[mask].copy()
MAX_PER_CATEGORY = 60

def take_first_n(group):
    return group.head(MAX_PER_CATEGORY)

df_uk = df_uk.groupby('categoryname').apply(take_first_n).reset_index(drop=True)
df_uk.reset_index(drop=True, inplace=True)

print("Проверка доступности изображений, подождите...")
df_uk['image_exists'] = df_uk['imgurl'].apply(url_points_to_image)
df = df_uk[df_uk['image_exists'] == True]

df_uk = df_uk.drop_duplicates()
category_counts = df_uk['categoryname'].value_counts()
df_uk = df_uk.reset_index(drop=True)
print(category_counts)
df_uk['price'] = df_uk['price'].astype(str)
df_uk['price'] = df_uk['price'] + '£'

df_uk['combined_features'] = (
    df_uk['title'] + ' ' +
    df_uk['categoryname'].astype(str) + ' ' +
    df_uk['price'].astype(str)
)

print(df_uk['categoryname'].unique())

sampled_df = df_uk.iloc[:1000].copy()
sampled_df.reset_index(drop=True, inplace=True)
sentences = sampled_df['combined_features'].tolist()
sampled_sentences = sentences[:1000]
embeddings = model.encode(sampled_sentences, show_progress_bar=True, convert_to_tensor=False)
cosine_sim = cosine_similarity(embeddings, embeddings)
final_df_uk = df_uk.copy()

if __name__ == '__main__':
    print(device)
    lighting_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('coffee & espresso machines')].shape[0]
    print(f"Количество товаров в категории 'Coffee & Espresso Machines': {lighting_count}")
    car_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('car & motorbike')].shape[0]
    print(f"Количество товаров в категории 'Car & Motorbike': {car_count}")
    smart_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('smartwatches')].shape[0]
    print(f"Количество товаров в категории 'Smartwatches': {smart_count}")
    bin_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('binoculars, telescopes & optics')].shape[0]
    print(f"Количество товаров в категории 'Binoculars, Telescopes & Optics': {bin_count}")
    clocks_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('clocks')].shape[0]
    print(f"Количество товаров в категории 'Clocks': {clocks_count}")
    gps_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('gps, finders & accessories')].shape[0]
    print(f"Количество товаров в категории 'GPS, Finders & Accessories': {gps_count}")
    hifi_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('hi-fi receivers & separates')].shape[0]
    print(f"Количество товаров в категории 'Hi-Fi Receivers & Separates': {hifi_count}")
    tele_count = df_uk[df_uk['categoryname'].str.lower().str.startswith('telephones, voip & accessories')].shape[0]
    print(f"Количество товаров в категории 'Telephones, VoIP & Accessories': {tele_count}")
    pass