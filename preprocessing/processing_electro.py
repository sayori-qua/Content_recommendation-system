import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, KNNBasic, accuracy, SVD, SVDpp
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from surprise.model_selection import KFold
import requests
from sqlalchemy import create_engine
from models.model_loader import model

device = "cuda" if torch.cuda.is_available() else "cpu"
engine = create_engine('')

def url_points_to_image(url):
    try:
        response = requests.head(url, timeout=3)
        content_type = response.headers.get('Content-Type', '')
        return response.status_code == 200 and 'image' in content_type
    except Exception as e:
        return False

df = pd.read_sql('SELECT * FROM electro_products', engine)

df = df.dropna()
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['user_id'] = df['user_id'].str.split(',')
df['product_name'] = df['product_name'].fillna('')
df['review_content'] = df['review_content'].fillna('')
df['about_product'] = df['about_product'].fillna('')
df['category'] = df['category'].fillna('')
df['actual_price'] = df['actual_price'].fillna('')
df['about_product'] = df['about_product'].str.replace('|', '', regex=False)

print("Проверка доступности изображений, подождите...")
df['image_exists'] = df['img_link'].apply(url_points_to_image)
df = df[df['image_exists'] == True]

df_exploded = df.explode('user_id')
df_exploded = df_exploded.drop_duplicates(subset=['product_name', 'about_product'])
df_exploded = df_exploded.reset_index(drop=True)

df_exploded['actual_price'] = df_exploded['actual_price'].str.replace('₹', '', regex=False)
df_exploded['actual_price'] = df_exploded['actual_price'].str.split(',').str[0]
df_exploded['actual_price'] = df_exploded['actual_price'].str.split('.').str[0]
df_exploded['actual_price'] = df_exploded['actual_price'].astype(int) * 0.0085
df_exploded['actual_price'] = df_exploded['actual_price'].astype(int)
df_exploded['actual_price'] = df_exploded['actual_price'].astype(str)
df_exploded['actual_price'] = df_exploded['actual_price'] + '£'

exclude_prefixes = ['home', 'health', 'car', 'music', 'toys']
pattern = '|'.join(f'^{prefix}' for prefix in exclude_prefixes)
df_exploded = df_exploded[~df_exploded['category'].str.contains(pattern, case=False, na=False)]
df_exploded['product_id'] = df_exploded.index
category_counts = df_exploded['category'].value_counts()
print(category_counts)

le = LabelEncoder()
df_collab = df_exploded[['product_id', 'user_id', 'rating', 'category']].copy()
df_collab['product_id'] = le.fit_transform(df_collab['product_id'])
df_collab['user_id'] = le.fit_transform(df_collab['user_id'])
df_collab['category'] = le.fit_transform(df_collab['category'])
df_collab['rating'] = pd.to_numeric(df_collab['rating'], errors='coerce')

df_collab = df_collab.drop_duplicates(subset=['user_id', 'product_id'])
mean_rating = df_collab['rating'].mean()
df_collab['rating'] = df_collab['rating'].fillna(mean_rating)

#data для contenta
df_exploded['combined_features'] = (
    df_exploded['product_name'] + ' ' +
    df_exploded['review_content'] + ' ' +
    df_exploded['actual_price'] + ' ' +
    df_exploded['about_product'] + ' ' +
    df_exploded['category'].astype(str)
)

sentences = df_exploded['combined_features'].tolist()
embeddings = model.encode(sentences, show_progress_bar=True, convert_to_tensor=False)
cosine_sim = cosine_similarity(embeddings, embeddings) #косинусное сходство

final_df_exploded = df_exploded.copy()

if __name__ == '__main__':
    print(device)
    #data для collaba
    data = Dataset.load_builtin('ml-100k')
    data.raw_ratings = [
        (str(row.user_id), str(row.product_id), row.rating, None)
        for _, row in df_collab.iterrows()
    ]

    #анализ
    group1 = df.groupby('category')['rating'].mean().sort_values(ascending=False)
    top_20_categories = group1.head(20)

    plt.figure(figsize=(12, 6))
    top_20_categories.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Mean rating in category')
    plt.xlabel('Category')
    plt.ylabel('Mean rating')
    plt.ylim(4.46, 4.68)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    train_collab, test_collab = train_test_split(data, test_size = 0.2, random_state = 42)

    sim_options = {
        "name": "cosine",
        "user_based": True,
    }

    models_for_collab = {'SVD' : SVD(),
                         'SVD++' : SVDpp(),
                         'KNN' : KNNBasic(sim_options=sim_options)}

    res_collab_cross = {}

    kf = KFold(n_splits=6)
    for models_name, model in models_for_collab.items():
        for train_data, test_data in kf.split(data):
            model.fit(train_data)
            predictions = model.test(test_data)
            rmse_cross = accuracy.rmse(predictions)
            mae_cross = accuracy.mae(predictions)
            res_collab_cross[models_name] = {'RMSE_cross_val': rmse_cross,
                                            'MAE_cross_val': mae_cross
                                            }
    print(f"Results for data_collab with cross validation: {res_collab_cross}")

    res_collab = {}
    for models_name, model in models_for_collab.items():
        model.fit(train_collab)
        predictions = model.test(test_collab)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        res_collab[models_name] = {'RMSE' : rmse,
                                   'MAE' : mae
                                   }
    print(f"Results for data_collab: {res_collab}") #best это SVD++
    best_model = models_for_collab['SVD++'] #тк историю взаимодействий я получить не смогу, то как пример буду использовать контентную фильтрацию
    pass