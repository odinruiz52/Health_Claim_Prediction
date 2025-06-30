import pandas as pd
import joblib

# Load the final model
model = joblib.load('final_claim_model.pkl')

# Load the encoded dataset
df = pd.read_csv('insurance_encoded.csv')

def predict_claim_status(data):
    prediction = model.predict(data)
    return 'Denied' if prediction[0] == 1 else 'Approved'

def get_user_input():
    print("Select an option:")
    print("1. Predict using a specific data point from the dataset")
    print("2. Input new data for prediction")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        index = int(input("Enter the row index of the data point (e.g., 0 for the first row): "))
        selected_data = df.iloc[[index], :-1]  # Exclude the 'claim_status' column
        print("Selected Data Point:")
        print(selected_data)
        status = predict_claim_status(selected_data)
        print(f"Predicted Claim Status: {status}")

    elif choice == '2':
        new_data = pd.DataFrame({
            'age': [int(input("Age: "))],
            'sex': [int(input("Sex (0 for female, 1 for male): "))],
            'bmi': [float(input("BMI: "))],
            'children': [int(input("Children: "))],
            'smoker': [int(input("Smoker (0 for no, 1 for yes): "))],
            'region': [int(input("Region (0, 1, 2, or 3): "))],
            'charges': [float(input("Charges: "))]
        })
        print("Input Data:")
        print(new_data)
        status = predict_claim_status(new_data)
        print(f"Predicted Claim Status: {status}")

    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    get_user_input()
