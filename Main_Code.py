import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

#filepath
data_file_path = 'data.txt'

#ticker tape to variable conversion
colspecs = [
    (51, 53),   #education
    (58, 59),   #sex
    (59, 61),   #race
    (63, 66),   #age
    (76, 77),   #marital Status
    (141, 144), #ICD_Code
]

#gives names to variables
column_names = [
    'Education',
    'Sex',
    'Race',
    'Age',
    'Marital_Status',
    'ICD_Code'
]

#reads data
df = pd.read_fwf(
    data_file_path,
    colspecs=colspecs,
    names=column_names,
    dtype=str
)

#strips whitespace
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

#converts fields into numeric values if needed. Used in data processing
numeric_fields = [
    'Education',
    'Sex',
    'Race',
    'Age',
    'Marital_Status',
    'ICD_Code'
]

for field in numeric_fields:
    df[field] = pd.to_numeric(df[field], errors='coerce')

#code mapping for sex
sex_mapping = {1: 'Male', 2: 'Female'}
df['Sex'] = df['Sex'].map(sex_mapping)

#code mapping for marital status
marital_status_mapping = {
    1: 'Never married, single',
    2: 'Married',
    3: 'Widowed',
    4: 'Divorced',
    8: 'Marital status not on certificate',
    9: 'Marital status not stated',
}

df['Marital_Status'] = df['Marital_Status'].map(marital_status_mapping)

#code mapping for education
education_mapping = {
    0: 'No formal education',
    1: '1 year of elementary school',
    2: '2 years of elementary school',
    3: '3 years of elementary school',
    4: '4 years of elementary school',
    5: '5 years of elementary school',
    6: '6 years of elementary school',
    7: '7 years of elementary school',
    8: '8 years of elementary school',
    9: '1 year of high school',
    10: '2 years of high school',
    11: '3 years of high school',
    12: '4 years of high school',
    13: '1 year of college',
    14: '2 years of college',
    15: '3 years of college',
    16: '4 years of college',
    17: '5 or more years of college',
    99: 'Not stated',
}

df['Education'] = df['Education'].astype(float).astype('Int64')
df['Education'] = df['Education'].map(education_mapping)

#splits marital status into male and female
df['Marital_Status_Sex'] = df['Marital_Status'] + ' - ' + df['Sex']

#ICD range map function
def map_icd9_code_with_ranges(code, mapping):
    code = str(code).strip().upper()
    if pd.isnull(code) or code == '':
        return 'Unknown ICD-9 Code'
    code = code.lstrip('0')  
    code_no_dot = code.replace('.', '') 

    
    if code in mapping:
        return mapping[code]

   
    for key in mapping:
        key_parts = key.split(',')
        for part in key_parts:
            part = part.strip().upper()
            part_no_dot = part.replace('.', '')
            if '-' in part:
                start_str, end_str = part.split('-')
                start_str, end_str = start_str.strip(), end_str.strip()
                start_no_dot = start_str.replace('.', '').lstrip('0')
                end_no_dot = end_str.replace('.', '').lstrip('0')
                if start_no_dot <= code_no_dot <= end_no_dot:
                    return mapping[key]
            else:
                if code_no_dot == part_no_dot.lstrip('0'):
                    return mapping[key]
    return 'Unknown ICD-9 Code'

#ICD code mapping
icd_code_mapping = {
   '004,006': 'Shigellosis and amebiasis',
   '007-009': 'Certain other intestinal infections',
   '010-018': 'Tuberculosis',
   '010-012': 'Tuberculosis of respiratory system',
   '013-018': 'Other tuberculosis',
   '033': 'Whooping cough',
   '034-035': 'Streptococcal sore throat, scarlatina, and erysipelas',
   '036': 'Meningococcal infection',
   '038': 'Septicemia',
   '045': 'Acute poliomyelitis',
   '055': 'Measles',
   '070': 'Viral hepatitis',
   '090-097': 'Syphilis',
   '001-003,005,020-032,037,039-041,046-054,056-066,071-088,098-139': 'All other infectious and parasitic diseases',
   '140-208': 'Malignant neoplasms, including neoplasms of lymphatic and hematopoietic tissues',
   '140-149': 'Malignant neoplasms of lip, oral cavity, and pharynx',
   '150-159': 'Malignant neoplasms of digestive organs and peritoneum',
   '160-165': 'Malignant neoplasms of respiratory and intrathoracic organs',
   '174-175': 'Malignant neoplasm of breast',
   '179-187': 'Malignant neoplasms of genital organs',
   '188-189': 'Malignant neoplasms of urinary organs',
   '170-173,190-199': 'Malignant neoplasms of all other and unspecified sites',
   '204-208': 'Leukemia',
   '200-203': 'Other malignant neoplasms of lymphatic and hematopoietic tissues',
   '210-239': 'Benign neoplasms, carcinoma in situ, and neoplasms of uncertain behavior and unspecified nature',
   '250': 'Diabetes mellitus',
   '260-269': 'Nutritional deficiencies',
   '280-285': 'Anemias',
   '320-322': 'Meningitis',
   '390-448': 'Major cardiovascular diseases',
   '390-398,402,404-429': 'Diseases of heart',
   '390-398': 'Rheumatic fever and rheumatic heart disease',
   '402': 'Hypertensive heart disease',
   '404': 'Hypertensive heart and renal disease',
   '410-414': 'Ischemic heart disease',
   '410': 'Acute myocardial infarction',
   '411': 'Other acute and subacute forms of ischemic heart disease',
   '413': 'Angina pectoris',
   '412,414': 'Old myocardial infarction and other forms of chronic ischemic heart disease',
   '424': 'Other diseases of endocardium',
   '415-423,425-429': 'All other forms of heart disease',
   '401,403': 'Hypertension with or without renal disease',
   '430-438': 'Cerebrovascular diseases',
   '431-432': 'Intracerebral and other intracranial hemorrhage',
   '4340,4349': 'Cerebral thrombosis and unspecified occlusion of cerebral arteries',
   '4341': 'Cerebral embolism',
   '430,433,435-438': 'All other and late effects of cerebrovascular diseases',
   '440': 'Atherosclerosis',
   '441-448': 'Other diseases of arteries, arterioles, and capillaries',
   '466': 'Acute bronchitis and bronchiolitis',
   '480-487': 'Pneumonia and influenza',
   '480-486': 'Pneumonia',
   '487': 'Influenza',
   '490-496': 'Chronic obstructive pulmonary diseases and allied conditions',
   '490-491': 'Bronchitis, chronic and unspecified',
   '492': 'Emphysema',
   '493': 'Asthma',
   '494-496': 'Other chronic obstructive pulmonary diseases and allied conditions',
   '531-533': 'Ulcer of stomach and duodenum',
   '540-543': 'Appendicitis',
   '550-553,560': 'Hernia of abdominal cavity and intestinal obstruction without mention of hernia',
   '571': 'Chronic liver disease and cirrhosis',
   '574-575': 'Cholelithiasis and other disorders of gallbladder',
   '580-589': 'Nephritis, nephrotic syndrome, and nephrosis',
   '580-581': 'Acute glomerulonephritis and nephrotic syndrome',
   '582-583,587': 'Chronic glomerulonephritis, nephritis and nephropathy, and renal sclerosis',
   '584-586,588-589': 'Renal failure, disorders resulting from impaired renal function, and small kidney of unknown cause',
   '590': 'Infections of kidney',
   '600': 'Hyperplasia of prostate',
   '630-676': 'Complications of pregnancy, childbirth, and the puerperium',
   '630-638': 'Pregnancy with abortive outcome',
   '640-676': 'Other complications of pregnancy, childbirth, and the puerperium',
   '740-759': 'Congenital anomalies',
   '760-779': 'Certain conditions originating in the perinatal period',
   '767-769': 'Birth trauma, intrauterine hypoxia, birth asphyxia, and respiratory distress syndrome',
   '760-766,770-779': 'Other conditions originating in the perinatal period',
   '780-799': 'Symptoms, signs, and ill-defined conditions',
   'E800-E949': 'Accidents and adverse effects',
   'E810-E825': 'Motor vehicle accidents',
   'E800-E807,E826-E949': 'All other accidents and adverse effects',
   'E950-E959': 'Suicide',
   'E960-E978': 'Homicide and legal intervention',
   'E980-E999': 'All other external causes'
}

df['Cause_of_Death'] = df['ICD_Code'].apply(lambda x: map_icd9_code_with_ranges(x, icd_code_mapping))

#post process data cleaning
df = df.dropna(subset=['Age'])
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df = df.dropna(subset=['Age'])

#preps features and target variable
X = df.drop(['Age', 'ICD_Code', 'Marital_Status', 'Sex'], axis=1)
y = df['Age']

X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

#linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

#ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

#gradient boosting
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mse)
    print(f"{name} - MSE: {mse:.2f}, RMSE: {rmse:.2f} R² Score: {r2:.2f}")

evaluate_model('Linear Regression', y_test, y_pred_lr)
evaluate_model('Ridge Regression ', y_test, y_pred_ridge)
evaluate_model('Gradient Boosting', y_test, y_pred_gbr)

#scatter plot for each side by side
models = {
    'Linear Regression': y_pred_lr,
    'Ridge Regression': y_pred_ridge,
    'Gradient Boosting Regression': y_pred_gbr,
}

plt.figure(figsize=(18, 5))
for idx, (name, y_pred) in enumerate(models.items(), 1):
    plt.subplot(1, 3, idx)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title(name)
plt.tight_layout()
plt.show()

#importance for each model
importances_lr = pd.Series(lr.coef_, index=X_train.columns).sort_values()
importances_ridge = pd.Series(ridge.coef_, index=X_train.columns).sort_values()
importances_gbr = pd.Series(gbr.feature_importances_, index=X_train.columns).sort_values()


#gradient boosting plot
plt.figure(figsize=(12, 8))
importances_gbr.tail(10).plot(kind='barh')
plt.title('Top 10 Features - Gradient Boosting Regression')
plt.xlabel('Feature Importance Score')
plt.show()

#ridge regression plot
plt.figure(figsize=(12, 8))
importances_ridge.tail(10).plot(kind='barh')
plt.title('Top 10 Features - Ridge Regression')
plt.xlabel('Feature Importance Score')
plt.show()

#linear regression plot
plt.figure(figsize=(12, 8))
importances_lr.tail(10).plot(kind='barh')
plt.title('Top 10 Features - Linear Regression')
plt.xlabel('Coefficient Value')
plt.show()

#top 10 causes of death plot
cause_counts = df['Cause_of_Death'].value_counts().head(10)
cause_counts.plot(kind='barh')
plt.title('Top 10 Causes of Death')
plt.xlabel('Number of Deaths')
plt.gca().invert_yaxis()
plt.show()

#indices sort to make data look nice :)
y_test_sorted = np.array(y_test).flatten()
y_pred_sorted = np.array(y_pred_lr).flatten()
indices = np.argsort(y_test_sorted)
y_test_sorted = y_test_sorted[indices]
y_pred_sorted = y_pred_sorted[indices]
plt.figure(figsize=(12, 8))

#cool lookin bar plot
bar_width = 0.4
x = np.arange(len(y_test_sorted))
plt.bar(x - bar_width / 2, y_test_sorted, width=bar_width, color='blue', label='Actual Age')
plt.bar(x + bar_width / 2, y_pred_sorted, width=bar_width, color='orange', label='Predicted Age')
plt.xlabel('Sample Index')
plt.ylabel('Age')
plt.title('Actual vs Predicted Age of Death')
plt.legend()
plt.tight_layout()
plt.show()

#residuals plot
residuals = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title('Residuals (Actual - Predicted Age)')
plt.xlabel('Sample Index')
plt.ylabel('Residual')
plt.show()



