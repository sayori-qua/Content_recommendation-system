from preprocessing.processing_electro import cosine_sim, sampled_df

def recommend_products_for_eprod_by_id(product_idx, original_product_id, cosine_sim=cosine_sim, sampled_df=sampled_df, top_n=4):
    sim_scores = list(enumerate(cosine_sim[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended = []
    seen_names = set()
    original_name = sampled_df.iloc[product_idx]['product_name']
    for i, score in sim_scores:
        if i == product_idx:
            continue
        prod = sampled_df.iloc[i]
        name_clean = prod['product_name']
        if name_clean == original_name or score > 0.98:
            continue
        if name_clean in seen_names:
            continue
        seen_names.add(name_clean)
        recommended.append({
            'product_name': prod['product_name'],
            'product_id': sampled_df.index[i],
            'category': prod['category'],
            'about_product': prod['about_product'],
            'similarity_score': round(score, 4),
            'img_link': prod.get('img_link', '')
        })
        if len(recommended) >= top_n:
            break
    return recommended

def get_product_index_by_name(product_name, df):
    matches = df[df['product_name'] == product_name].index
    return matches[0] if len(matches) > 0 else None

product_name = "Computer"
product_idx = get_product_index_by_name(product_name, sampled_df)
if product_idx is not None and product_idx < len(sampled_df):
    recommendations = recommend_products_for_eprod_by_id(product_idx, product_idx)
    print(recommendations)