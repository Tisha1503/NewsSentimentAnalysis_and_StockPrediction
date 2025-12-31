# FinBERT Real-Time Market Intelligence Engine 

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/NLP-FinBERT-green.svg)](https://huggingface.co/ProsusAI/finbert)

### link: https://newssentimentanalysisandstockprediction-7yowdoctitimbajxabxbjr.streamlit.app/

##  Project Overview
This project is a quantitative NLP pipeline designed to extract, analyze, and correlate real-time financial sentiment with equity price action. By leveraging FinBERT, a BERT-based model pre-trained on financial corpora (10-Ks, 10-Qs, and earnings calls), the system identifies market signals from unstructured news data that general-purpose NLP models typically miss.

The engine doesn't just visualize sentiment; it validates it using Pearson Correlation Analysis, providing a mathematical proof of the relationship between news polarity and market returns.

---

## Technical Architecture



### **1. NLP Pipeline (FinBERT)**
* **Domain Specificity:** Uses `ProsusAI/finbert` via the Hugging Face Transformers library to handle financial nuance (e.g., recognizing "interest rate hikes" as a bearish signal rather than general "growth").
* **Sentiment Filtering:** Designed to handle "Financial Neutrality," a common trait in factual corporate reporting that often skews general sentiment models.

### **2. Quantitative Analytics**
* **Data Harmonization:** Synchronizes disparate data frequencies (irregular news timestamps vs. structured daily market intervals) using Pandas resampling.
* **Statistical Validation:** Calculates the **Pearson Correlation Coefficient ($r$)** to measure the linear relationship between daily aggregated sentiment and next-day price returns.
* **Metric Calculation:** Computes Period Returns, Volatility Bias, and Confidence Scores for every inference.

### **3. Dynamic Visualization**
* **Dual-Axis Convergence:** A custom Plotly implementation that overlays daily sentiment bars against stock price lines to visualize lead-lag effects.
* **Polarity Distribution:** Real-time breakdown of sentiment density (Positive vs. Negative vs. Neutral).

---

## Installation & Usage

### **Prerequisites**
* Python 3.12+
* NewsAPI Key (Get one at [newsapi.org](https://newsapi.org/))

### **Setup**
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Tisha1503/NewsSentimentAnalysis_and_StockPrediction.git](https://github.com/Tisha1503/NewsSentimentAnalysis_and_StockPrediction.git)
   cd NewsSentimentAnalysis_and_StockPrediction
### **install dependencies:**
pip install -r requirements.txt

### **Run the Application:**
streamlit run app.py

### **Sample Output & Methodology**
During testing with blue-chip equities like MCD (McDonald's Corp), the system demonstrated:

High Pearson Correlation: Observed correlations as high as 0.74, indicating a strong alignment between news sentiment and price movement.

Neutrality Accuracy: Correctly identified ~75% of news as Neutral, reflecting the factual nature of standard corporate press releases and avoiding "sentiment inflation."

### **Project Structure**
app.py: The Streamlit dashboard and UI logic.

news_analysis.py: The core engine containing the NLP pipeline and statistical correlation methods.

requirements.txt: Environment dependencies.

.gitignore: Prevents sensitive API keys and cache files from being uploaded.

### **Author: Tisha Thakkar**
