## Titanic Dataset Analysis

This Jupyter Notebook (titanicDataset.ipynb) contains an exploratory data analysis (EDA) and feature engineering process applied to the classic Titanic dataset. The goal is to understand the data, identify patterns, and prepare the data for potential machine learning models to predict passenger survival.

### Libraries Used

*   **NumPy:** For numerical computations.
*   **Pandas:** For data manipulation and analysis.
*   **Seaborn:** For data visualization.
*   **Matplotlib:** For creating plots and graphs.

### Data Source

The dataset used is 'Titanic-Dataset.csv', which contains information about passengers aboard the Titanic, including demographics, travel details, and survival status.

### Data Loading and Exploration

1.  **Importing Libraries:** The necessary libraries are imported.
2.  **Importing the Data:** The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`.
3.  **Initial Exploration:**
    *   `dataset.head()`: Displays the first few rows of the dataset to get a quick overview of the data.
    *   `dataset.describe()`: Provides descriptive statistics of the numerical columns, such as count, mean, standard deviation, etc.
    *   Survival rate is calculated based on Pclass, Sex and Embarked

### Feature Engineering

Several new features were engineered to potentially improve the predictive power of the data:

*   **Sex Encoding:** The 'Sex' column (categorical) is converted into numerical data by replacing 'male' with 1 and 'female' with 0.
*   **Age Banding:**
    *   The 'Age' column (numerical) is categorized into ordinal data by creating age bands.
        *   A new column, 'AgeBand', is created based on these bins.
        *   The survival rate is calculated for each age band.
*   **Alone Feature:**
    *   A new feature 'is_alone' is created to identify passengers traveling alone. It combines 'SibSp' (number of siblings/spouses aboard) and 'Parch' (number of parents/children aboard).
    *   If both 'SibSp' and 'Parch' are 0, the passenger is marked as alone (1), otherwise not alone (0).
    *   The survival rate is calculated for passengers that are alone or not alone.
*   **Final Data Selection:**
    *   The final data is selected and the name column is removed.

### Next Steps

This notebook provides a foundation for further analysis and modeling. Potential next steps include:

*   **Model Building:** Using the preprocessed data to train machine learning models for survival prediction.
*   **Hyperparameter Tuning:** Optimizing model parameters to improve performance.
*   **Cross-Validation:** Evaluating model performance using cross-validation techniques.
