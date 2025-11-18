from preprocessing.processing_uk_data import cosine_sim, sampled_df

def recommend_products_for_uk_data_by_id(product_idx, original_product_id, cosine_sim=cosine_sim, sampled_df=sampled_df, top_n=4):
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended = []
    seen_names = set()
    original_name = sampled_df.iloc[product_idx]['title']
    max_reviews = sampled_df['reviews'].max()
    max_stars = 5.0
    for i, score in sim_scores:
        if i == product_idx:
            continue
        prod = sampled_df.iloc[i]
        name_clean = prod['title']
        if name_clean == original_name or score > 0.95:
            continue
        if name_clean in seen_names:
            continue
        seen_names.add(name_clean)
        review_norm = prod['reviews'] / max_reviews if max_reviews > 0 else 0
        star_norm = prod['stars'] / max_stars if max_stars > 0 else 0
        weighted_score = (score + 0.1 * star_norm + 0.05 * review_norm) #дополнительно взвешиваем для оптимизации рекомендаций
        recommended.append({
            'product_id': sampled_df.index[i],
            'title': prod['title'],
            'categoryname': prod['categoryname'],
            'similarity_score': round(score, 4),
            'stars': prod['stars'],
            'reviews': prod['reviews'],
            'imgurl': prod.get('imgurl', ''),
            'weighted_score': round(weighted_score, 4)
        })
        if len(recommended) >= top_n:
            break
    recommended = sorted(recommended, key=lambda x: x['weighted_score'], reverse=True)
    return recommended

def get_product_index_by_name(product_name, df):
    matches = df[df['title'] == product_name].index
    return matches[0] if len(matches) > 0 else None

product_name = "Echo Show 8 | 2nd generation (2021 release), HD smart display with Alexa and 13 MP camera | Charcoal"
product_idx = get_product_index_by_name(product_name, sampled_df)
if product_idx is not None and product_idx < len(sampled_df):
    recommendations = recommend_products_for_uk_data_by_id(product_idx, product_idx)
    print(recommendations)