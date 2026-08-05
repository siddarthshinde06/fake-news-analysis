# 📰 Fake News Detection System

A Flask-based web application that classifies news articles as **Real** or **Fake** using a machine learning model trained with TF-IDF vectorization and Logistic Regression.

![Fake News](image.jpg)

## Overview

Fake News Detection System lets users paste a news headline or article into a simple web interface. The text is cleaned, transformed with a saved TF-IDF vectorizer, and passed to a trained Logistic Regression model, which predicts whether the article is **Real News** or **Fake News**.

## Features

- Simple, single-page web interface for pasting and checking news text
- Text cleaning pipeline (lowercasing, URL/HTML/punctuation removal, etc.) applied before prediction
- Pretrained TF-IDF vectorizer and Logistic Regression model loaded via `joblib`
- Color-coded result display (green for real, red for fake)
- Jupyter notebook included showing the full model training process

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn (TF-IDF Vectorizer, Logistic Regression), joblib
- **Frontend:** HTML, CSS
- **Data handling:** pandas

## Model Performance

The Logistic Regression model was trained on a labeled dataset of real and fake news articles (`true.csv` / `fake.csv`) split 75/25 for training and testing. On the held-out test set it achieved:

| Metric | Score |
|---|---|
| Accuracy | ~98.6% |
| Precision (avg) | 0.99 |
| Recall (avg) | 0.99 |
| F1-score (avg) | 0.99 |

## Project Structure

```
fake-news-analysis/
├── app.py            # Flask application (routes, prediction logic)
├── app.ipynb         # Notebook used to clean data, train, and export the model
├── index.html         # Front-end page rendered by Flask
├── style.css           # Styling for the web page
├── image.jpg           # Banner/preview image
├── lr_model.jb         # Trained Logistic Regression model (joblib)
├── vectorizer.jb        # Fitted TF-IDF vectorizer (joblib)
├── LICENSE
└── README.md
```

> **Note:** Flask expects HTML templates in a `templates/` folder and static assets (like `style.css`) in a `static/` folder. If you run into `TemplateNotFound` or missing-style errors, create these folders and move `index.html` into `templates/` and `style.css` into `static/`.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/siddarthshinde06/fake-news-analysis.git
   cd fake-news-analysis
   ```

2. Install the required dependencies
   ```bash
   pip install flask scikit-learn pandas joblib
   ```

3. Run the application
   ```bash
   python app.py
   ```

4. Open your browser and go to 

### Usage

1. Paste a news headline or article into the text box.
2. Click **Check News**.
3. The predicted label (Real News / Fake News) is displayed below the form.

## How It Works

1. **Preprocessing:** Input text is cleaned (lowercased, stripped of URLs, HTML tags, punctuation, and digits).
2. **Vectorization:** The cleaned text is transformed into numerical features using a pretrained TF-IDF vectorizer (`vectorizer.jb`).
3. **Prediction:** The feature vector is passed to a trained Logistic Regression model (`lr_model.jb`), which outputs a class: `1` for real news, `0` for fake news.
4. **Display:** The Flask app renders the result back to the user with a color-coded label.

## Training the Model

The full training pipeline is available in `app.ipynb` and includes:

- Loading and labeling the real (`true.csv`) and fake (`fake.csv`) news datasets
- Cleaning and merging the text data
- Splitting into training and test sets
- Fitting a `TfidfVectorizer` on the training text
- Training a `LogisticRegression` classifier
- Evaluating with a classification report
- Exporting the vectorizer and model with `joblib`

To retrain the model, place `true.csv` and `fake.csv` in the project directory and run the notebook. we did not upload the csv file this are available on kaggle 

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Siddarth Shinde**
