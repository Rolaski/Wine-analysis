# Wine Dataset Analysis

An interactive application for analyzing and exploring the Wine Dataset from UCI using Python and data analysis libraries (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit).

## Project Description

This application allows you to:
- Read and analyze data from the Wine Dataset
- Perform statistical analysis (min, max, median, mode, standard deviation, etc.)
- Determine correlations between features
- Extract subtables and manipulate data
- Visualize data (dependency graphs, histograms, correlation graphs)
- Perform analysis using classification models, clustering, and association rules
- Interact with data through a user-friendly GUI

## Project Structure

```
wine_analysis_app/
│
├── data/                    # Dataset files
│ ├── wine.data              # Raw data
│ ├── wine.names             # Dataset description
│ └── index                  # Index file
│
├── src/                     # Core functionality modules
│ ├── __init__.py
│ ├── data_loader.py         # Data loading module
│ ├── data_manipulator.py    # Data manipulation module
│ ├── statistical_analyzer.py # Statistical analysis module
│ ├── data_visualizer.py     # Visualization module
│ ├── ml_modeler.py          # ML modeling module
│ └── utils.py               # Helper functions
│
├── pages/                   # Application pages
│ ├── __init__.py
│ ├── data_overview.py       # Data overview page
│ ├── statistical_analysis.py # Statistical analysis page
│ ├── data_manipulation.py   # Data manipulation page
│ ├── visualization.py       # Visualization page
│ └── ml_modeling.py         # ML modeling page
│
├── components/              # Reusable UI components
│ ├── __init__.py
│ ├── descriptions.py        # Text descriptions for UI
│ ├── ui_helpers.py          # UI helper functions
│ └── sidebar.py             # Sidebar navigation component
│
├── wine-analysis-desktop/   # Electron desktop app (created by setup script)
│
├── requirements.txt         # Python dependencies
├── setup_electron_app.py    # Script to create desktop application
└── app.py                   # Main Streamlit application
```

## Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`
- For desktop version: Node.js and npm

## Running the Application

### Option 1: Web Application (Streamlit)

This is the simplest way to run the application as a web interface:

1. **Clone the repository and set up a virtual environment**:
   ```bash
   git clone https://github.com/yourusername/wine-analysis.git
   cd wine-analysis
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Unix/MacOS
   source venv/bin/activate
   ```

2. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

4. The application will be available in your browser at `http://localhost:8501`

### Option 2: Desktop Application (Electron)

This creates a standalone desktop application with its own window:

1. **Set up the Electron project** (if not already done):
   ```bash
   python setup_electron_app.py
   ```

2. **Configure the environment**:
   ```bash
   cd wine-analysis-desktop
   
   # On Windows
   setup_env.bat
   
   # On Unix/MacOS
   ./setup_env.sh
   ```

3. **Run the application in development mode**:
   ```bash
   # Make sure you're in the wine-analysis-desktop directory
   npm run dev
   ```

4. **Build a production version** (creates installable files):
   ```bash
   # Make sure you're in the wine-analysis-desktop directory
   
   # For Windows
   npm run build-win
   
   # For macOS
   npm run build-mac
   
   # For Linux
   npm run build-linux
   ```

5. Installer files will be available in the `wine-analysis-desktop/dist/` directory

## Features

### Data Overview
- View basic dataset information
- View data samples
- Descriptive statistics
- Class distribution

### Statistical Analysis
- Calculate basic statistical measures
- Examine correlations between features
- Identify outliers
- Statistical tests

### Data Manipulation
- Select features and rows
- Replace values
- Handle missing data
- Scaling and standardization
- Binary encoding

### Visualization
- Histograms
- Box plots
- Scatter plots (2D and 3D)
- Correlation matrices
- Feature pair plots
- Parallel coordinate plots

### ML Modeling
- Classification (KNN, SVM, Random Forest)
- Clustering (K-Means, DBSCAN)
- Association rules (Apriori)
- Results visualization
- Model parameter selection and optimization

## Troubleshooting

### Web Application Issues
- If you encounter `ModuleNotFoundError`, make sure all required packages are installed: `pip install -r requirements.txt`
- If the app doesn't launch, check if port 8501 is already in use by another application

### Desktop Application Issues
- Make sure Node.js and npm are installed
- If the application closes immediately, run it from the command line to see error messages
- Check if Python is properly installed and available in the PATH
- If building fails, make sure you have the necessary build tools installed:
  - Windows: Visual Studio Build Tools
  - macOS: Xcode Command Line Tools
  - Linux: build-essential package


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine) for the Wine Dataset
- Streamlit for the web framework
- Electron for desktop application capabilities