
# **Titanic Dataset Analysis**  

This Jupyter Notebook (**titanicDataset.ipynb**) explores the Titanic dataset through **Exploratory Data Analysis (EDA)** and **feature engineering**. The goal is to understand patterns in the data and prepare it for **machine learning models** to predict passenger survival.  

---

## **1. Libraries Used**  

- **NumPy** – For numerical computations.  
- **Pandas** – For data manipulation and analysis.  
- **Seaborn** – For data visualization.  
- **Matplotlib** – For creating plots and graphs.  

---

## **2. Data Overview**  

**Dataset:** `Titanic-Dataset.csv`  
This dataset contains information about passengers aboard the Titanic, including demographics, travel details, and survival status.  

### **Data Loading & Exploration**  

1. **Importing Libraries** – Required Python libraries are loaded.  
2. **Loading the Data** – The dataset is read into a Pandas DataFrame using `pd.read_csv()`.  
3. **Initial Exploration:**  
   - `dataset.head()` – Displays the first few rows.  
   - `dataset.describe()` – Provides summary statistics.  
   - **Survival rates** are analyzed based on **Pclass, Sex, and Embarked**.  

---

## **3. Feature Engineering**  

To improve predictive performance, new features were engineered:  

### **A. Encoding Categorical Data**  
- **Sex Encoding** – Converted 'Sex' into numerical values:  
  - `'male' → 1, 'female' → 0`  

### **B. Creating New Features**  
- **Age Binning**  
  - 'Age' is grouped into predefined age bands.  
  - A new column, `'AgeBand'`, is created.  
  - Survival rates are calculated for each age group.  

- **Is Alone Feature**  
  - A new column `'is_alone'` is added.  
  - If a passenger has no **siblings/spouses (`SibSp`)** and no **parents/children (`Parch`)**, they are considered **alone (`1`)**, otherwise **not alone (`0`)**.  
  - Survival rates are compared between passengers traveling alone vs. with family.  

### **C. Handling Categorical Features**  
- **Embarked Encoding**  
  - Dummy variables are created using `pd.get_dummies()`.  
  - The categorical `'Embarked'` feature is converted into multiple numerical columns (`Embarked_C`, `Embarked_Q`, `Embarked_S`).  
  - The original `'Embarked'` column is dropped.  

---

## **4. Model Building (Commented Out)**  

The notebook includes commented-out code for **machine learning models**, including an **Artificial Neural Network (ANN)** and an **XGBoost Classifier**.  

### **A. Artificial Neural Network (ANN)**  

A **Sequential ANN model** is defined using **Keras**, with:  
- Multiple dense layers with **ReLU activation**  
- Output layer with **sigmoid activation** (for binary classification)  
- **Adam optimizer** and **binary cross-entropy loss**  



### **B. XGBoost Classifier**  

An **XGBoost Classifier** is defined with:  
- **Max Depth = 4**  
- **Learning Rate = 0.1**  
- **100 Trees (Estimators)**  

