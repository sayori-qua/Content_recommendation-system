from preprocessing.processing_electro import cosine_sim, df_exploded

def recommend_products_for_eprod(product_name, cosine_sim=cosine_sim, df=df_exploded, top_n=4):
    df = df.copy()
    df['product_name_clean'] = df['product_name'].str.lower().str.strip()
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
    for i, score in sim_scores:
        if i == idx:
            continue
        prod = df.iloc[i]
        name_clean = prod['product_name_clean']
        if name_clean == original_name or score > 0.98:
            continue
        if name_clean in seen_names:
            continue
        seen_names.add(name_clean)
        recommended.append({
            'product_name': prod['product_name'],
            'product_id': df.index[i],
            'category': prod['category'],
            'about_product': prod['about_product'],
            'similarity_score': round(score, 4),
            'img_link': prod.get('img_link', '')
        })
        if len(recommended) >= top_n:
            break
    return recommended

def evaluate_content_model_for_eprod(df, recommend_func, top_n=4):
    df = df.copy()
    df['category'] = df['category'].astype(str)
    correct = 0
    total = 0
    sample_products = df.sample(n=20, random_state=42)
    for _, row in sample_products.iterrows():
        product_name = row['product_name']
        true_category = str(row['category']).strip().lower()
        recommendations = recommend_func(product_name)
        if not recommendations:
            continue
        predicted_categories = [str(item['category']).strip().lower() for item in recommendations]
        count_same_category = sum([1 for cat in predicted_categories if cat == true_category])
        correct += count_same_category
        total += top_n
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0
    print(f"% Рекомендаций из той же категории: {accuracy:.2%}")
    return accuracy

print(recommend_products_for_eprod("Computer"))
evaluate_content_model_for_eprod(df_exploded, recommend_func=recommend_products_for_eprod)