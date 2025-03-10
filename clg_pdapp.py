import pandas as pd
import numpy as np
import streamlit as st
import mysql.connector
from mysql.connector import errorcode
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('\n', '_').str.upper()
    df = df.dropna(subset=['TUITION_FEE'])
    df['TUITION_FEE'] = pd.to_numeric(df['TUITION_FEE'], errors='coerce')

    # Fill missing values in rank columns with a large number
    rank_columns = [col for col in df.columns if 'BOYS' in col or 'GIRLS' in col]
    for col in rank_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].max() + 1)
    
    return df

# Train a machine learning model
def train_model(df):
    # Prepare the dataset
    features = df[['BRANCH_CODE', 'DIST', 'TUITION_FEE']].copy()
    label_encoders = {}
    for col in features.columns:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        label_encoders[col] = le
    
    rank_columns = [col for col in df.columns if 'BOYS' in col or 'GIRLS' in col]
    X = features
    y = df[rank_columns].min(axis=1) <= df[rank_columns].min(axis=1).median()  # Binary classification: 1 if rank is below median, else 0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    st.write("Model Accuracy:", accuracy)
    st.write("ROC-AUC Score:", roc_auc)
    
    return model, label_encoders

# Make predictions based on user inputs
def predict_colleges(model, label_encoders, user_inputs):
    encoded_inputs = user_inputs.copy()
    for col in encoded_inputs.columns:
        encoded_inputs[col] = label_encoders[col].transform(encoded_inputs[col])
    
    predictions = model.predict_proba(encoded_inputs)[:, 1]  # Probability of admission
    return predictions

# Filter data based on user inputs
def filter_data(df, expected_rank, category, districts, courses):
    # Filter by district and course
    filtered_df = df[(df['BRANCH_CODE'].isin(courses)) & (df['DIST'].isin(districts))]
    
    # Filter by rank category
    filtered_df = filtered_df[filtered_df[category] >= expected_rank]

    return filtered_df

# Save filtered data to MySQL database
def save_to_mysql(df):
    db_config = {
        'user': 'root',
        'password': 'Faisal123@',
        'host': '127.0.0.1',
        'port': 3306,
        'database': 'Faisal'
    }

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS filtered_colleges (
            Institute_Name VARCHAR(255),
            Place VARCHAR(255),
            Dist VARCHAR(255),
            Branch_Name VARCHAR(255),
            Tuition_Fee FLOAT
        )
        """
        cursor.execute(create_table_query)
        
        insert_query = """
        INSERT INTO filtered_colleges (Institute_Name, Place, Dist, Branch_Name, Tuition_Fee)
        VALUES (%s, %s, %s, %s, %s)
        """
        for _, row in df.iterrows():
            cursor.execute(insert_query, (row['INSTITUTE_NAME'], row['PLACE'], row['DIST'], row['BRANCH_NAME'], row['TUITION_FEE']))
        
        conn.commit()
        cursor.close()
        conn.close()
        st.success("Data successfully saved to the database!")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            st.error("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            st.error("Database does not exist")
        else:
            st.error(err)

# Main function
def main():
    st.title('TSEAMCET 2023 College Predictor')

    # Load the dataset
    file_path = 'C:\\New folder\\Copy of 01_TSEAMCET_2023_FirstPhase_LastRanks.csv'
    df = load_data(file_path)
    
    # Train the machine learning model
    model, label_encoders = train_model(df)
    
    # User inputs
    st.sidebar.header('User Input Parameters')
    
    expected_rank = st.sidebar.number_input("Enter your expected rank:", min_value=1, max_value=200000, value=100000)
    category = st.sidebar.selectbox("Select your category:", options=["OC_BOYS", "OC_GIRLS", "BC_A_BOYS", "BC_A_GIRLS", "BC_B_BOYS", "BC_B_GIRLS", "BC_C_BOYS", "BC_C_GIRLS", "BC_D_BOYS", "BC_D_GIRLS", "BC_E_BOYS", "BC_E_GIRLS", "SC_BOYS", "SC_GIRLS", "ST_BOYS", "ST_GIRLS", "EWS_GEN_OU", "EWS_GIRLS_OU"])
    
    districts = st.sidebar.multiselect("Select preferred districts:", options=df['DIST'].unique())
    courses = st.sidebar.multiselect("Select preferred courses:", options=df['BRANCH_CODE'].unique())
    
    if st.sidebar.button('Show Results'):
        filtered_df = filter_data(df, expected_rank, category, districts, courses)
        if not filtered_df.empty:
            # Prepare user inputs for prediction
            user_inputs = filtered_df[['BRANCH_CODE', 'DIST', 'TUITION_FEE']].copy()
            
            predictions = predict_colleges(model, label_encoders, user_inputs)
            filtered_df = filtered_df.assign(ADMISSION_PROBABILITY=predictions)
            
            sorted_df = filtered_df.sort_values(by='ADMISSION_PROBABILITY', ascending=False)
            top_colleges = sorted_df.head(6)
            
            st.write("Top colleges based on your preferences:")
            st.write(top_colleges[['INSTITUTE_NAME', 'PLACE', 'DIST', 'BRANCH_NAME', 'TUITION_FEE', 'ADMISSION_PROBABILITY']])
            
            st.bar_chart(top_colleges.set_index('INSTITUTE_NAME')['ADMISSION_PROBABILITY'])
            
            # Save the filtered results to the database
            save_to_mysql(top_colleges)
        else:
            st.write("No colleges found based on the given criteria.")

if __name__ == "__main__":
    main()
