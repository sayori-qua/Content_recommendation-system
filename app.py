import pandas as pd
from flask import Flask, request, session, redirect, url_for
import torch
from flask import render_template
from recommenders.rec_func_for_eprod import recommend_products_for_eprod
from recommenders.rec_func_for_uk_data import recommend_products_for_uk_data
from preprocessing.processing_electro import final_df_exploded as df_exploded
from preprocessing.processing_uk_data import final_df_uk as df_uk
from elasticsearch import Elasticsearch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

app = Flask(__name__)

app.secret_key = 'sayori_qua_78'

categories = df_exploded['category'].unique().tolist()
products_electro = df_exploded['product_name'].tolist()
img_links_electro = df_exploded['img_link'].tolist()
info_about_products = df_exploded['about_product'].tolist()
uk_categories = df_uk['categoryname'].str.split('|').str[0].unique().tolist()

es = Elasticsearch(
    "http://elasticsearch:9200",
    verify_certs=False,
    request_timeout=30,
    sniff_on_start=False,
    sniff_before_requests=False,
    retry_on_timeout=True,
    max_retries=3,

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

df_all = pd.concat([df_1, df_2], ignore_index=True)

if es is not None:
    try:
        if not es.indices.exists(index="products"):
            for _, row in df_all.iterrows():
                es.index(index='products', document={
                    'product_name': row['product_name'],
                    'category': row['category'],
                    'img_link': row['img_link'] if 'img_link' in row and pd.notna(row['img_link']) else None,
                    "dataset": row["dataset"],
                    "product_id": int(row["product_id"])
                })
            print("Indexing completed.")
        else:
            print("Index 'products' already exists. Skipping indexing.")
    except Exception as e:
        print(f"Failed to manage index: {e}")
        es = None
else:
    print("Elasticsearch is not available. Skipping indexing.")
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
    df = df_exploded
    product_row = df[df.index == product_id]
    if product_row.empty:
        return "Product not found", 404
    product_row = product_row.iloc[0]
    product_name = product_row['product_name']
    product_img = product_row['img_link']
    product_desc = product_row['about_product']
    product_category = product_row['category']
    recommendations = recommend_products_for_eprod(product_name)
    return render_template(
        "product.html",
        name=product_name,
        image=product_img,
        description=product_desc,
        product_id=product_id,
        recommendations=recommendations,
        category=product_category,
        dataset="Electro"
    )

@app.route("/product/uk/<int:product_id>")
def product_page_uk(product_id):
    if product_id < 0 or product_id >= len(df_uk):
        return "Product not found", 404
    product_name = df_uk.iloc[product_id]['title']
    product_img = df_uk.iloc[product_id]['imgurl']
    product_category = df_uk.iloc[product_id]['categoryname']
    recommendations = recommend_products_for_uk_data(product_name)
    return render_template(
        "product.html",
        name=product_name,
        image=product_img,
        product_id=product_id,
        recommendations=recommendations,
        category=product_category,
        dataset="UK"
    )

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

    # Применяем фильтрацию
    filtered_df['price_numeric'] = filtered_df.apply(get_valid_price, axis=1)
    filtered_df = filtered_df[filtered_df['price_numeric'].notna()]

    print(f"📊 Found {len(filtered_df)} products for '{prefix}' → mapped to '{real_category}'")

    if filtered_df.empty:
        return f"No products found for category '{prefix}'", 404

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

@app.route("/add_to_cart/<string:dataset>/<int:product_id>")
def add_to_cart(dataset, product_id):
    if 'cart' not in session:
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

@app.route("/remove_from_cart/<int:index>")
def remove_from_cart(index):
    cart = session.get('cart', [])
    if 0 <= index < len(cart):
        del cart[index]
        session['cart'] = cart
    return redirect(url_for('view_cart'))

@app.context_processor
def inject_categories():
    return dict(
        electro_display_categories=ELECTRO_DISPLAY_CATEGORIES,
        uk_display_categories=uk_categories
    )


@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return redirect(url_for("index"))
    if es is None:#поиск без es
        filtered_df = df_all[
            df_all['product_name'].str.contains(query, case=False, na=False) |
            df_all['category'].str.contains(query, case=False, na=False)
            ].copy()
        prices = [] #добавляем цены
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
    else:
        try:
            es_query = {
                "multi_match": {
                    "query": query,
                    "fields": ["product_name^2", "category"]
                }
            }
            response = es.search(index="products", query=es_query)
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                price = None
                if source.get("dataset") == "Electro":
                    prod_id = source.get("product_id")
                    if prod_id is not None and 0 <= prod_id < len(df_exploded):
                        price = df_exploded.iloc[prod_id].get('actual_price')
                elif source.get("dataset") == "UK":
                    prod_id = source.get("product_id")
                    if prod_id is not None and 0 <= prod_id < len(df_uk):
                        price = df_uk.iloc[prod_id].get('price')
                if price is not None and price != 'N/A':
                    try:
                        clean_price = str(price).replace('£', '').replace('$', '').replace(',', '').strip()
                        price_num = float(clean_price)
                        if price_num <= 0:
                            continue
                    except (ValueError, TypeError):
                        continue
                results.append({
                    "product_name": source.get("product_name"),
                    "category": source.get("category"),
                    "img_link": source.get("img_link"),
                    "dataset": source.get("dataset"),
                    "product_id": source.get("product_id"),
                    "price": price
                })
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
            def is_valid_price(price_str): #фильтрация по цене
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
    app.run(host="0.0.0.0", port=5000)