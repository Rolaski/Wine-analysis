# Wine Dataset Analysis

Interactive application for analyzing and exploring the Wine Dataset from UCI using Python and data analysis libraries (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit).

## Project description

The application allows:
- Reading and analyzing data from a CSV file
- Performing statistical analysis (min, max, median, mode, standard deviation, etc.)
- Determining correlations between features
- Extracting subtables and manipulating data
- Visualizing data (dependency graphs, histograms, correlation graphs)
- Performing analysis using classification models, grouping and association rules
- Interactive data analysis via GUI

## Project structure

```
wine_analysis_app/
│
├── data/
│ ├── wine.data
│ ├── wine.names
│ └── index
│
├── src/
│ ├── __init__.py
│ ├── data_loader.py # Data loading module
│ ├── data_manipulator.py # Data manipulation module
│ ├── statistical_analyzer.py # Statistical analysis module
│ ├── data_visualizer.py # Visualization module
│ ├── ml_modeler.py # ML modeling module
│ └── utils.py # Helper functions
│
├── requirements.txt
├── README.md
└── app.py # Main Streamlit application
```

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt` file

## Installation

1. Clone the repository:
```
git clone https://github.com/Rolaski/Wine-analysis.git
cd wine_analysis_app
```

2. Create and activate a virtual environment (optional):
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install the required libraries:
```
pip install -r requirements.txt
```

4. Prepare the data:
- Download the Wine Dataset from the UCI Machine Learning Repository
- Place the `wine.data`, `wine.names` and `Index` files in the `data/` directory

## Running the application

To run the application, execute the following command in the root directory project:

```
streamlit run app.py
```

The application will be available in the browser at `http://localhost:8501`.

## Features

### Data Overview
- Viewing Basic Dataset Information
- Viewing a Data Sample
- Descriptive Statistics
- Class Distribution

### Statistical Analysis
- Calculating Basic Statistical Measures
- Examining Correlations Between Features
- Identifying Outliers
- Statistical Tests

### Data Manipulation
- Selecting Features and Rows
- Replacing Values
- Handling Missing Data
- Scaling and Standardization
- Binary Encoding

### Visualization
- Histograms
- Box Plots
- Scatter Plots (2D and 3D)
- Correlation Matrices
- Feature Pair Plots
- Parallel Coordinate Plots

### ML Modeling
- Classification (KNN, SVM, Random Forest)
- Clustering (K-Means, DBSCAN)
- Association Rules (Apriori)
- Results visualization
- Model parameter selection and optimization

## Authors

- [Jakub Jakubowski](https://github.com/Rolaski)
- [Kacper Bułaś](https://github.com/bolson1313)