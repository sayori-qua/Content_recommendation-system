import pandas as pd
from flask import Flask, request, session, redirect, url_for
import torch
from flask import render_template
from recommenders.rec_func_for_eprod import recommend_products_for_eprod_by_id
from recommenders.rec_func_for_uk_data import recommend_products_for_uk_data_by_id
from preprocessing.processing_electro import final_df_exploded as df_exploded
from preprocessing.processing_uk_data import final_df_uk as df_uk
from elasticsearch import Elasticsearch
import os
from preprocessing.processing_electro import sampled_df as sampled_electro
from preprocessing.processing_uk_data import sampled_df as sampled_uk

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

app = Flask(__name__)

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-for-dev-only')

categories = df_exploded['category'].unique().tolist()
products_electro = df_exploded['product_name'].tolist()
img_links_electro = df_exploded['img_link'].tolist()
info_about_products = df_exploded['about_product'].tolist()
uk_categories = df_uk['categoryname'].str.split('|').str[0].unique().tolist()

es = Elasticsearch(
    "http://elasticsearch:9200",
    verify_certs=False, #не проверяем ssl сертификаты
    request_timeout=30, #ожидание ответа от сервера
    sniff_on_start=False, #не сканируем кластер при инициализации
    sniff_before_requests=False, #не сканируем кластер перед каждым запросом
    retry_on_timeout=True, #клиент должен повторить запрос если он превысил таймаут
    max_retries=3, #максимальное количество попыток запроса
)

try:
    if es.ping():
        print("Elasticsearch is available")
    else:
        print("Elasticsearch ping returned False")
except Exception as e:
    print(f"Failed to connect to Elasticsearch: {e}")
    es = None

df_1 = df_exploded[['product_name', 'category', 'img_link']].copy()
df_1['dataset'] = 'Electro'
df_1['product_id'] = df_1.index

df_2 = df_uk[['title', 'categoryname', 'imgurl']].copy()
df_2 = df_2.rename(columns={'title': 'product_name', 'categoryname': 'category', 'imgurl': 'img_link'})
df_2['dataset'] = 'UK'
df_2['product_id'] = df_2.index

df_all = pd.concat([df_1, df_2], ignore_index=True) #объединенный датафрейм для es

id_mapping = {}
for idx, row in df_1.iterrows():
    id_mapping[idx] = ('Electro', idx) # idx - это original_index в df_exploded

for idx, row in df_2.iterrows():
    id_mapping[idx] = ('UK', idx) # idx - это original_index в df_uk

if es is not None:
    try:
        if not es.indices.exists(index="products"):
            for df_row_idx, row in df_all.iterrows():
                dataset, original_idx = id_mapping[df_row_idx] # получаем оригинальный dataset и индекс
                es.index(index='products', document={
                    'product_name': row['product_name'],
                    'category': row['category'],
                    'img_link': row['img_link'] if 'img_link' in row and pd.notna(row['img_link']) else None,
                    "dataset": row["dataset"],
                    "product_id": int(original_idx),
                    "original_df_index": int(df_row_idx)
                })
            print("Indexing completed.")
        else:
            print("Index 'products' already exists. Skipping indexing.")
    except Exception as e:
        print(f"Failed to manage index: {e}")
        es = None
else:
    print("Elasticsearch is not available. Skipping indexing.")

#отображение и обращение
ELECTRO_DISPLAY_CATEGORIES = {
    'TVs and accessories': 'Electronics',
    'Connection accessories': 'Computers',
    'Office tools': 'Office',
}

@app.route("/")
def index():
    selected_uk = df_uk['categoryname'].str.split('|').str[0].unique()[:12].tolist()
    return render_template(
        "home.html",
        electro_cats=ELECTRO_DISPLAY_CATEGORIES.keys(),
        uk_cats=selected_uk
    )
@app.route("/product/electro/<int:product_id>")
def product_page_electro(product_id):
    print(f"DEBUG Product Page Electro: Received product_id={product_id}")
    print(f"DEBUG Product Page Electro: df_exploded index range: {df_exploded.index.min()} to {df_exploded.index.max()}")
    product_row = df_exploded[df_exploded.index == product_id]
    if product_row.empty:
        print(f"DEBUG Product Page Electro: ERROR - product_id {product_id} not found in df_exploded index range.")
        return "Product not found", 404
    product_row = product_row.iloc[0]
    product_name = product_row['product_name']
    product_img = product_row['img_link']
    product_desc = product_row['about_product']
    product_category = product_row['category']
    product_price = product_row.get('actual_price', 'N/A')

    print(f"DEBUG Product Page Electro: Found product name: {product_name[:50]}...") # Вывести часть имени
    print(f"DEBUG Product Page Electro: Category: {product_category}, Price: {product_price}")

    match_in_sampled = sampled_electro[sampled_electro.index == product_id]
    if not match_in_sampled.empty:
        sampled_idx = match_in_sampled.index[0]
        print(f"DEBUG Product Page Electro: Found match in sampled_electro at sampled_idx {sampled_idx}")
        recommendations = recommend_products_for_eprod_by_id(sampled_idx, product_id)
    else:
        print(f"DEBUG Product Page Electro: No match found in sampled_electro for product_id {product_id}, skipping recommendations.")
        recommendations = []

    return render_template(
        "product.html",
        name=product_name,
        image=product_img,
        description=product_desc,
        product_id=product_id,
        recommendations=recommendations,
        category=product_category,
        dataset="Electro",
        price=product_price
    )

@app.route("/product/uk/<int:product_id>")
def product_page_uk(product_id):
    print(f"DEBUG Product Page UK: Received product_id={product_id}")
    print(f"DEBUG Product Page UK: df_uk index range: {df_uk.index.min()} to {df_uk.index.max()}")
    if product_id < 0 or product_id >= len(df_uk):
        print(f"DEBUG Product Page UK: ERROR - product_id {product_id} is out of bounds for df_uk (len={len(df_uk)}).")
        return "Product not found", 404
    product_row = df_uk.iloc[product_id]
    product_name = product_row['title']
    product_img = product_row['imgurl']
    product_category = product_row['categoryname']
    product_price = product_row.get('price', 'N/A')

    print(f"DEBUG Product Page UK: Found product name: {product_name[:50]}...") # Вывести часть имени
    print(f"DEBUG Product Page UK: Category: {product_category}, Price: {product_price}")

    match_in_sampled = sampled_uk[sampled_uk.index == product_id]
    if not match_in_sampled.empty:
        sampled_idx = match_in_sampled.index[0]
        print(f"DEBUG Product Page UK: Found match in sampled_uk at sampled_idx {sampled_idx}")
        recommendations = recommend_products_for_uk_data_by_id(sampled_idx, product_id)
    else:
        print(f"DEBUG Product Page UK: No match found in sampled_uk for product_id {product_id}, skipping recommendations.")
        recommendations = []

    return render_template(
        "product.html",
        name=product_name,
        image=product_img,
        product_id=product_id,
        recommendations=recommendations,
        category=product_category,
        dataset="UK",
        price=product_price
    )

#отображаем категории
@app.route("/category/electro/<string:prefix>")
def category_electro(prefix):
    real_category = ELECTRO_DISPLAY_CATEGORIES.get(prefix, prefix)
    filtered_df = df_exploded[
        df_exploded['category'].str.contains(real_category, case=False, na=False)
    ].copy()
    def get_valid_price(row): #фильтрация
        price_raw = row.get('actual_price', 'N/A')
        if pd.isna(price_raw) or price_raw == 'N/A':
            return None
        try:
            price_clean = str(price_raw).replace('£', '').replace('$', '').replace(',', '').strip()
            price_num = float(price_clean)
            return price_num if price_num > 0 else None
        except (ValueError, TypeError):
            return None
    #Применяем фильтрацию
    filtered_df['price_numeric'] = filtered_df.apply(get_valid_price, axis=1)
    filtered_df = filtered_df[filtered_df['price_numeric'].notna()]
    print(f"Found {len(filtered_df)} products for '{prefix}' → mapped to '{real_category}'")
    if filtered_df.empty:
        return f"No products found for category '{prefix}'", 404
    #пагинация если будем увеличивать количество товаров
    per_page = 200
    page = request.args.get('page', default=1, type=int)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = filtered_df.iloc[start:end]
    products_list = paginated_items.apply(
        lambda row: (row.name, row['product_name'], row['img_link'], row.get('actual_price', 'N/A')), axis=1
    ).tolist()
    total_pages = (len(filtered_df) + per_page - 1) // per_page

    return render_template(
        "all_products.html",
        products=products_list,
        category_name=prefix,
        dataset="Electro",
        current_page=page,
        total_pages=total_pages,
        prefix=prefix,
        endpoint="category_electro"
    )

@app.route("/category/uk/<string:prefix>")
def category_uk(prefix):
    filtered_df = df_uk[df_uk['categoryname'].str.lower().str.startswith(prefix.lower())].copy()
    def get_valid_price(row):
        price_raw = row.get('price', 'N/A')
        if pd.isna(price_raw) or price_raw == 'N/A':
            return None
        try:
            price_clean = str(price_raw).replace('£', '').replace('$', '').replace(',', '').strip()
            price_num = float(price_clean)
            return price_num if price_num > 0 else None
        except (ValueError, TypeError):
            return None
    filtered_df['price_numeric'] = filtered_df.apply(get_valid_price, axis=1) #применяем фильтрацию
    filtered_df = filtered_df[filtered_df['price_numeric'].notna()]
    if filtered_df.empty:
        return f"No products found for category starting with '{prefix}'", 404
    # пагинация если будем увеличивать количество товаров
    per_page = 200
    page = request.args.get('page', default=1, type=int)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = filtered_df.iloc[start:end]
    products_list = paginated_items.apply(
        lambda row: (row.name, row['title'], row['imgurl'], row.get('price', 'N/A')), axis=1
    ).tolist()
    total_pages = (len(filtered_df) + per_page - 1) // per_page

    return render_template(
        "all_products.html",
        products=products_list,
        category_name=prefix.capitalize(),
        dataset="UK",
        current_page=page,
        total_pages=total_pages,
        prefix=prefix,
        endpoint="category_uk"
    )
#добавление в корзину
@app.route("/add_to_cart/<string:dataset>/<int:product_id>")
def add_to_cart(dataset, product_id):
    if 'cart' not in session: #проверяем существует ил в сессии пользователя cart
        session['cart'] = []
    cart = session['cart']
    if dataset == "Electro":
        if product_id < 0 or product_id >= len(df_exploded):
            return "Product not found", 404
        product = df_exploded.iloc[product_id]
        cart_item = {
            'id': product_id,
            'name': product['product_name'],
            'image': product['img_link'],
            'price': product.get('actual_price', 'N/A'),
            'dataset': dataset
        }
    elif dataset == "UK":
        if product_id < 0 or product_id >= len(df_uk):
            return "Product not found", 404
        product = df_uk.iloc[product_id]
        cart_item = {
            'id': product_id,
            'name': product['title'],
            'image': product['imgurl'],
            'price': product.get('price', 'N/A'),
            'dataset': dataset
        }
    else:
        return "Invalid dataset", 400
    cart.append(cart_item)
    session['cart'] = cart
    return "Added to cart"

#корзина
@app.route("/cart")
def view_cart():
    cart = session.get('cart', [])
    total = 0.0
    for item in cart:
        price_str = item.get('price', '0')
        if isinstance(price_str, (int, float)):
            total += float(price_str)
        else:
            clean_price = str(price_str).replace('£', '').replace('$', '').replace(',', '').strip()
            try:
                total += float(clean_price)
            except ValueError:
                pass
    return render_template("cart.html", cart=cart, total=total)

#удаление товара из корзины по индексу
@app.route("/remove_from_cart/<int:index>")
def remove_from_cart(index):
    cart = session.get('cart', [])
    if 0 <= index < len(cart):
        del cart[index]
        session['cart'] = cart
    return redirect(url_for('view_cart'))

#автоматически передаем список категорий в каждый html файл
@app.context_processor
def inject_categories():
    return dict(
        electro_display_categories=ELECTRO_DISPLAY_CATEGORIES,
        uk_display_categories=uk_categories
    )

#поиск по товарам с использованием es и резервного метода на основе фильтрации
@app.route("/search")
def search():
    query = request.args.get("query", "").strip()#извлекаем запрос
    if not query:
        return redirect(url_for("index"))
    if not query:
        return redirect(url_for("index"))
    if es is None:  # если клиент ES не был инициализирован или недоступен
        print("Elasticsearch not available, using fallback search.")
        filtered_df = df_all[
            df_all['product_name'].str.contains(query, case=False, na=False) |
            df_all['category'].str.contains(query, case=False, na=False)
            ].copy()
        prices = []
        for _, row in filtered_df.iterrows():
            if row['dataset'] == 'Electro':
                if row['product_id'] < len(df_exploded):
                    price = df_exploded.iloc[row['product_id']].get('actual_price', 'N/A')
                else:
                    price = 'N/A'
            elif row['dataset'] == 'UK':
                if row['product_id'] < len(df_uk):
                    price = df_uk.iloc[row['product_id']].get('price', 'N/A')
                else:
                    price = 'N/A'
            else:
                price = 'N/A'
            prices.append(price)
        filtered_df['price'] = prices
        def is_valid_price(price_str):
            if pd.isna(price_str) or price_str == 'N/A':
                return False
            try:
                clean_price = str(price_str).replace('£', '').replace('$', '').replace(',', '').strip()
                price_num = float(clean_price)
                return price_num > 0
            except (ValueError, TypeError):
                return False
        filtered_df = filtered_df[filtered_df['price'].apply(is_valid_price)]
        results = filtered_df.head(50).to_dict('records')
        return render_template("search_results.html", query=query, results=results[:50])
    else:
        try:
            es_query = {
                "multi_match": { #тип запроса
                    "query": query,
                    "fields": ["product_name^2", "category"]
                }
            }
            response = es.search(index="products", query=es_query)
            results = []
            for hit in response["hits"]["hits"]:  #перебираем результаты в es
                source = hit["_source"]  #данные о товаре
                original_idx = source.get("product_id")
                dataset = source.get("dataset")
                print(f"DEBUG ES Search: Hit source: {source}")
                print(f"DEBUG ES Search: Extracted original_idx: {original_idx}, dataset: {dataset}")
                price = None
                if dataset == "Electro":
                    if 0 <= original_idx < len(df_exploded): # original_idx - это индекс в df_exploded
                        price = df_exploded.iloc[original_idx].get('actual_price')
                        print(f"DEBUG ES Search: Found price '{price}' for Electro product_id {original_idx} in df_exploded.")
                    else:
                        print(f"DEBUG ES Search: WARNING - original_idx {original_idx} for Electro is out of bounds for df_exploded (len={len(df_exploded)})")
                        continue
                elif dataset == "UK":
                    if 0 <= original_idx < len(df_uk):
                        price = df_uk.iloc[original_idx].get('price')
                        print(f"DEBUG ES Search: Found price '{price}' for UK product_id {original_idx} in df_uk.")
                    else:
                        print(f"DEBUG ES Search: WARNING - original_idx {original_idx} for UK is out of bounds for df_uk (len={len(df_uk)})")
                        continue
                else:
                    print(f"DEBUG ES Search: WARNING - Unknown dataset '{dataset}' for product_id {original_idx}")
                    continue
                if price is not None and price != 'N/A':
                    try:
                        clean_price = str(price).replace('£', '').replace('$', '').replace(',', '').strip()
                        price_num = float(clean_price)
                        if price_num <= 0:
                            print(f"DEBUG ES Search: Skipping product {original_idx} ({dataset}) due to non-positive price after cleaning: {price_num}")
                            continue
                    except (ValueError, TypeError):
                        print(f"DEBUG ES Search: Skipping product {original_idx} ({dataset}) due to invalid price format: {price}")
                        continue
                results.append({
                    "product_name": source.get("product_name"),
                    "category": source.get("category"),
                    "img_link": source.get("img_link"),
                    "dataset": source.get("dataset"),
                    "product_id": int(original_idx),
                    "price": price
                })

            print(f"DEBUG ES Search: Total results before template render: {len(results)}")
            if results:
                print(f"DEBUG ES Search: First result product_id: {results[0]['product_id']}, dataset: {results[0]['dataset']}, name: {results[0]['product_name'][:50]}...")
                print(f"DEBUG ES Search: Last result product_id: {results[-1]['product_id']}, dataset: {results[-1]['dataset']}, name: {results[-1]['product_name'][:50]}...")
            return render_template("search_results.html", query=query, results=results[:50])
        except Exception as e:
            print(f"Search failed: {e}")
            filtered_df = df_all[
                df_all['product_name'].str.contains(query, case=False, na=False) |
                df_all['category'].str.contains(query, case=False, na=False)
                ].copy()
            prices = []
            for _, row in filtered_df.iterrows():
                if row['dataset'] == 'Electro':
                    if row['product_id'] < len(df_exploded):
                        price = df_exploded.iloc[row['product_id']].get('actual_price', 'N/A')
                    else:
                        price = 'N/A'
                elif row['dataset'] == 'UK':
                    if row['product_id'] < len(df_uk):
                        price = df_uk.iloc[row['product_id']].get('price', 'N/A')
                    else:
                        price = 'N/A'
                else:
                    price = 'N/A'
                prices.append(price)
            filtered_df['price'] = prices
            def is_valid_price(price_str):
                if pd.isna(price_str) or price_str == 'N/A':
                    return False
                try:
                    clean_price = str(price_str).replace('£', '').replace('$', '').replace(',', '').strip()
                    price_num = float(clean_price)
                    return price_num > 0
                except (ValueError, TypeError):
                    return False
            filtered_df = filtered_df[filtered_df['price'].apply(is_valid_price)]
            results = filtered_df.head(50).to_dict('records')
            return render_template("search_results.html", query=query, results=results[:50])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) #запросы будут приходить через порт 5000