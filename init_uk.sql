CREATE TABLE IF NOT EXISTS amazon_products (
    asin TEXT,
    title TEXT,
    imgurl TEXT,
    productURL TEXT,
    stars DOUBLE PRECISION,
    reviews INTEGER,
    price DOUBLE PRECISION,
    isBestSeller BOOLEAN,
    boughtInLastMonth INTEGER,
    categoryName TEXT
);

COPY amazon_products FROM '/docker-entrypoint-initdb.d/amz_uk_processed_data.csv' DELIMITER ',' CSV HEADER;