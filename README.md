-----

````markdown
# 🏥 End-to-End Medical Insurance Cost Prediction

## 📌 Project Overview
This project aims to solve a critical challenge for insurance companies: **accurately estimating medical costs**. By analyzing a dataset of patient demographics and health metrics, we built a Machine Learning pipeline that predicts individual insurance charges with high precision. This solution automates the risk assessment process, allowing for fair and profitable premium setting.


## 💼 Business Problem
**The Challenge:**
Insurance companies traditionally rely on actuarial tables and manual estimation to set premiums. This process is:
1.  **Slow:** Manual review takes time.
2.  **Inconsistent:** Human error can lead to under-pricing (loss of revenue) or over-pricing (loss of customers).
3.  **Complex:** Factors like BMI and Smoking status interact in non-linear ways that are hard to calculate manually.

**The Solution:**
A Data Science pipeline that ingests user data and outputs a predicted dollar amount for medical charges, utilizing **Linear Regression** as the predictive engine.

---

## 📂 Dataset Description
The dataset contains **1,338 records** of beneficiaries. We used the following features to train our model:

| Feature | Data Type | Description |
| :--- | :--- | :--- |
| **Age** | Numerical | Age of the primary beneficiary (18-64). |
| **Sex** | Categorical | Gender of the contractor (Male/Female). |
| **BMI** | Numerical | Body Mass Index (kg / m^2). A key indicator of health. |
| **Children** | Numerical | Number of dependents covered by the plan. |
| **Smoker** | Categorical | Smoking status (Yes/No). **(Critical Feature)** |
| **Region** | Categorical | Residential area in the US (NE, NW, SE, SW). |
| **Charges** | Numerical | **Target Variable:** Individual medical costs billed by health insurance. |

---

## 🔍 Phase 1: Exploratory Data Analysis (EDA)
Before modeling, we performed a "health check" on the data to understand distributions and relationships.

### 1. Target Variable Diagnosis
We visualized the distribution of `charges`.
![Distribution of Charges](images/charges_dist.png)
* **Observation:** The data was highly **Right-Skewed**. Most people pay <$15,000, but a "long tail" of expensive cases extends to $60,000+.
* **Action:** We identified that Linear Regression would struggle with this skew. We planned a **Log Transformation** to normalize the target.

### 2. Feature Analysis (Outliers)
We checked `bmi` for anomalies.
![BMI Outliers](images/bmi_boxplot.png)
* **Observation:** Several patients had BMIs above 47, which are considered extreme outliers.
* **Action:** We decided to **Cap (Winsorize)** these values to prevent them from distorting the regression line.

### 3. Correlation Analysis
![Correlation Heatmap](images/heatmap.png)
* **Key Insight:** `Smoker` status has the strongest correlation with charges (approx 0.78). This confirms that smoking is the primary driver of cost, followed by Age and BMI.

---

## ⚙️ Phase 2: Data Preprocessing (The Pipeline)
Raw data cannot be fed into a mathematical model. We applied a rigorous preprocessing pipeline:

### 1. Outlier Handling
* **Method:** IQR (Interquartile Range) Method.
* **Execution:** Any BMI > 47 was capped at 47. This retains the data point but reduces its leverage on the model.

### 2. Normalization (Log Transform)
* **Method:** `np.log1p()`
* **Execution:** We transformed the `charges` column. This converted the skewed distribution into a **Normal (Bell Curve) Distribution**, satisfying the assumptions of Linear Regression.

### 3. Feature Encoding
Machine Learning models require numerical input.
* **Label Encoding:** Converted binary categories (`sex`, `smoker`) into `0` and `1`.
* **One-Hot Encoding:** Converted `region` into binary columns (`region_northwest`, `region_southeast`, etc.), utilizing `drop_first=True` to avoid Multicollinearity.

### 4. Feature Scaling
* **Method:** `StandardScaler`
* **Execution:** We scaled `age`, `bmi`, and `children` to have a Mean of 0 and Standard Deviation of 1.
* **Why?** `Age` (range 18-64) is numerically larger than `Region` (range 0-1). Scaling ensures the model treats all features fairly.

---

## 🤖 Phase 3: Model Training & Evaluation
We trained a **Linear Regression** model using Scikit-Learn.

### Training Process
1.  **Split:** 80% Training Data / 20% Testing Data.
2.  **Algorithm:** Ordinary Least Squares (OLS) Linear Regression.
3.  **Coefficients:** The model learned that **Smoking** adds the highest weight to the cost, confirming our EDA findings.

### Performance Metrics
We evaluated the model on the 20% Test Set.
*Note: Since we trained on Log charges, we applied `np.expm1()` to convert predictions back to real dollars.*

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | **0.78** | The model explains **78%** of the variance in medical costs. |
| **MAE** | **$4,150** | On average, predictions are within ~$4k of the actual bill. |
| **RMSE** | **$5,900** | Penalizes large errors more heavily (useful for detecting high-cost misses). |

---

## 🧪 Phase 4: Prediction System
We developed a robust function to allow for manual testing. This function replicates the entire preprocessing pipeline for new, single inputs.

```python
# Example of the internal logic
def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    # 1. Create DataFrame
    # 2. Encode Categorical Variables
    # 3. Scale Numerical Variables (using the training scaler)
    # 4. Predict (Log Scale)
    # 5. Convert to Real Dollars (np.expm1)
    return predicted_price
````

### Example Usage

**Input:**

  * Age: 35
  * Sex: Male
  * BMI: 30
  * Children: 2
  * Smoker: Yes
  * Region: Southeast

**Model Output:**

> 💰 **Predicted Insurance Charges: $32,450.20**

-----

## 🔮 Future Improvements

While the Linear Regression model performs well (78% accuracy), further improvements can be made:

1.  **Interaction Features:** Explicitly creating a `BMI * Smoker` feature, as high BMI is significantly more expensive for smokers than non-smokers.
2.  **Advanced Models:** Implementing **Random Forest** or **XGBoost** to capture non-linear relationships, which could boost accuracy to **85%+**.
3.  **Deployment:** Building a Streamlit web app to allow non-technical users to utilize the tool.

-----

## 💻 How to Run This Project

### Installation  
1. Clone the repository:  
    ```bash
    git clone https://github.com/sudipta9749/Medical-Insurance-Cost-Prediction-Using-LR.git
    cd Medical-Insurance-Cost-Prediction-Using-LR
    ```  
2. (Optional but recommended) Create and activate a virtual environment:  
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # on Windows: venv\Scripts\activate
    ```  
3. Install required packages:  
    ```bash
    pip install -r requirements.txt
   
    ```

-----

- **Customization**: You can easily modify parameters (e.g. test-train split ratio) or extend the model (e.g. try Ridge/Lasso regression) by updating the code in `Notebook`.

## 📈 Results & What You’ll Get  
After running the project end-to-end, you’ll get:  
- Cleaned and processed dataset ready for modeling.  
- A trained Linear Regression model.  
- Evaluation metrics such as RMSE, MAE, R² describing how well the model predicts insurance cost.  
- (Optionally) Insight into feature importance — which attributes impact cost most (e.g. smoker status, BMI, age).  

Feel free to plug in your own data to test predictions.

## 🛠️ Tech Stack / Libraries Used  
- Python 3.x  
- pandas — for data manipulation  
- numpy — for numerical operations  
- scikit-learn — for building and evaluating the Linear Regression model  
- (Optional) Jupyter Notebook — for interactive data exploration & visualization

## 📝 Future Work / Roadmap  
- Extend to more sophisticated models (ensemble methods, regularized regression).  
- Add feature engineering (e.g. interaction terms, polynomial features).  
- Build a simple web UI to input personal details and get a cost estimate.  
- Add data visualization of results and exploratory data analysis (EDA).  
- Incorporate cross-validation and hyperparameter tuning for robust performance.  

*Author: [Sudipta Biswas]*

