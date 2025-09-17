<div align="center">
  <h1>Content recommendation system with analysis</h1>
</div>
<div align="center">
  <h3>Written by @sayori_qua</h3>
</div>

>Exploring product trends and patterns in the Amazon UK electronics marketplace using real datasets from Kaggle. The project analyzes thousands of real Amazon UK product listings focusing on electronics categories including Lighting, Smart Speakers, Cameras, Torches, Coffee & Espresso Machines, Car & Motorbike products, Smartwatches, Binoculars, Telescopes & Optics, Clocks, GPS devices, Hi-Fi Receivers & Separates, and Telephones. Using advanced data processing techniques, we examined pricing trends, review patterns, category performance, and consumer behavior patterns to uncover valuable market insights. The heart of the project is an advanced content-based recommendation system powered by sentence transformers and cosine similarity algorithms. By combining product titles, descriptions, prices, categories, and review data into semantic embeddings, the system can intelligently suggest similar products to users. The recommendation engine uses the all-mpnet-base-v2 model to create rich semantic representations of products, enabling accurate matching even for products with different wording but similar features. The system also incorporates weighted scoring that considers star ratings and review counts to prioritize high-quality recommendations. The entire project is packaged into a comprehensive Docker containerized application that makes deployment effortless. The multi-container architecture includes separate services for the Flask web application, PostgreSQL databases for UK and electronics products, Elasticsearch for advanced search capabilities, and all necessary dependencies pre-configured. This Docker implementation ensures consistent behavior across different environments, easy scaling, and simplified maintenance. Users can launch the complete application stack with a single docker-compose command, making it accessible to both developers and end users.

---

## What Analyzed
- **Product Categories**: Lighting, Smart Speakers, Cameras, Smartwatches, and more
- **Key Metrics**: Pricing trends, review patterns, and category performance
- **Data Points**: Thousands of real Amazon UK product listings

## Key Insights
### Top Performing Categories
- Smart Speakers & Home Assistants
- Camera Equipment & Accessories  
- Smartwatches & Wearables
- Coffee & Espresso Machines
- Automotive Electronics

### Price Analysis
- **Highest Average Prices**: Hi-Fi Receivers, Professional Cameras
- **Best Value Categories**: Basic Lighting, Simple Smart Speakers
- **Premium Segment**: Smartwatches, GPS Devices

## Machine Learning Features
### Content-Based Recommendation System
- Uses product descriptions, prices, and categories
- Sentence transformers for semantic similarity
- Smart product matching algorithm

### Collaborative Filtering
- User behavior pattern analysis
- SVD++ recommendation engine
- Cross-validation for accuracy

## Technical Stack
- **Data Processing**: Python, Pandas, NumPy
- **ML Models**: Sentence Transformers, Scikit-learn, Surprise
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Flask, PostgreSQL, Elasticsearch
- **Deployment**: Docker containers

## Live Features
- Cross-dataset product recommendations
- Elasticsearch-powered search
- Shopping cart functionality
- Responsive web interface
- Real-time data visualization

## Sample Visualizations
![Mean Price by Category](graphics/mean_price.png)
*Average pricing across top electronics categories*

![Review Distribution](graphics/Mean_quantity.png)
*Review volume analysis by product type*
## Docker Deployment
The entire application is containerized using Docker for easy deployment and scalability. The multi-container architecture includes the web application, PostgreSQL databases, and Elasticsearch search engine.

### Docker Setup
# Clone the repository
git clone your-repo-url
'''bash
cd your-project-directory
'''
# Build and start all services
'''bash
docker-compose up -d
'''
Access the application at http://localhost:5000

## Getting Started
```bash
# Install dependencies
pip install -r requirements.txt
```
# Run EDA analysis
python eda/first_amazon_dataset.py

# Start web application
python app.py
