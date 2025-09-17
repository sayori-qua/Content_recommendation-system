from preprocessing.processing_uk_data import cosine_sim, sampled_df

def recommend_products_for_uk_data(product_name, cosine_sim=cosine_sim, sampled_df=sampled_df, top_n=4):
    df = sampled_df.copy()
    df['product_name_clean'] = df['title'].str.lower().str.strip()
    product_name_clean = product_name.lower().strip()
    matches = df[df['product_name_clean'].str.contains(product_name_clean, na=False, regex=False)]
    if matches.empty:
        print(f"Товар '{product_name}' не найден.")
        return []
    match_pos = matches.index[0]
    idx = df.index.get_loc(match_pos)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended = []
    seen_names = set()
    original_name = df.iloc[idx]['product_name_clean']
    max_reviews = sampled_df['reviews'].max()
    max_stars = 5.0
    for i, score in sim_scores:
        if i == idx:
            continue
        prod = df.iloc[i]
        name_clean = prod['product_name_clean']
        if name_clean == original_name or score > 0.95:
            continue
        if name_clean in seen_names:
            continue
        seen_names.add(name_clean)
        review_norm = prod['reviews'] / max_reviews if max_reviews > 0 else 0
        star_norm = prod['stars'] / max_stars if max_stars > 0 else 0
        weighted_score = (score + 0.1 * star_norm + 0.05 * review_norm)
        recommended.append({
            'product_id': df.index[i],
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

def evaluate_content_model_for_uk_data(df, recommend_func, top_n=4):
    df = sampled_df.copy()
    df['categoryname'] = df['categoryname'].astype(str)
    correct = 0
    total = 0
    sample_products = df.sample(n=20, random_state=42)
    for _, row in sample_products.iterrows():
        product_name = row['title']
        true_category = str(row['categoryname']).strip().lower()
        recommendations = recommend_func(product_name)
        if not recommendations:
            continue
        predicted_categories = [str(item['categoryname']).strip().lower() for item in recommendations]
        count_same_category = sum([1 for cat in predicted_categories if cat == true_category])
        correct += count_same_category
        total += top_n
    if total>0:
        accuracy = correct / total
    else:
        accuracy = 0
    print(f"% Рекомендаций из той же категории: {accuracy:.2%}")
    return accuracy

print(recommend_products_for_uk_data("Echo Show 8 | 2nd generation (2021 release), HD smart display with Alexa and 13 MP camera | Charcoal"))
accuracy = evaluate_content_model_for_uk_data(sampled_df, recommend_products_for_uk_data)