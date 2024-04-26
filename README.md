# ieee-cs-ai-ml-heart-
//google colab python code for prediction of a heart disease
https://colab.research.google.com/drive/1jnJIVlbjOeGWpssNdcNTIiI4OeLa_AjH?usp=sharing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#to read the dataset
df = pd.read_csv('https://drive.google.com/uc?id=1CEql-OEexf9p02M5vCC1RDLXibHYE9Xz')
#to split the features (X) and target variable (y)
X = df.drop('target', axis=1)  
y = df['target']  
#to split the dataset into training and testing set using train-test-split (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# training the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# accepting the user input for data to predict
print("Enter patient details:")
age = int(input("Age: "))
sex = int(input("Sex (1 for female, 0 for male): "))
cp = int(input("Chest pain type (0-3):1.0- No chest pain. You don't feel any discomfort or pain in your chest area. \n 2.1-It usually occurs with physical activity or stress and goes away with rest. \n 2.It usually occurs with physical activity or stress and goes away with rest. \n 3.The pain is usually sharp or stabbing and doesn't worsen with physical activity.- "))
trestbps = int(input("Resting blood pressure(90/60 to 120/80mm/hg): "))
chol = int(input("Serum cholesterol (mg/dl)(5-7): "))
fbs = int(input("Fasting blood sugar > 120 mg/dl (1 for true, 0 for false): "))
restecg = int(input("Resting electrocardiographic results (0-2):1.0-normal: "))
thalach = int(input("Maximum heart rate achieved:(220-age): "))
exang = int(input("Exercise induced angina (1 for yes, 0 for no): "))
oldpeak = float(input("ST depression induced by exercise relative to rest: "))
slope = int(input("Slope of the peak exercise ST segment (0-2): "))
ca = int(input("Number of major vessels (0-3-0: No major vessels showing significant blockages.\n 1: One major vessel showing significant blockage. \n 2: Two major vessels showing significant blockages. \n 3: Three major vessels showing significant blockages.) colored by fluoroscopy: "))
thal = int(input("Thalassemia (0-3): "))
# Creating a DataFrame of the users input
new_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                        columns=X.columns)
# Predicting output for the new data
predicted_prob = model.predict_proba(new_data)[0, 1]  # Probability of having a heart disease
if predicted_prob > 0.5:
    print("The patient is likely to have heart disease.")
else:
    print("The patient is not likely to have heart disease.")
