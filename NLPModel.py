from project.NLPTrainer import *

print(" ")
if __name__ == "__main__":
    train = input("Do you want to train the model? (Y/N): ")
    if (train == "Y"):
        df = load_dataset()
        X, y, pipeline, label_encoder, all_symptoms = preprocess_dataset(df)
        print(f"Feature matrix shape: {X.shape}, Label matrix shape: {y.shape}")
    while True:
        text = input("Enter your symptoms (e.g., 'I feel tired and have a fever') or 'quit' to exit: ")
        if text.lower() == 'quit':
            break
        
        # Optional inputs
        age = input("Enter age (default 30): ")
        age = float(age) if age.strip() else None
        
        gender = input("Enter gender (Male/Female, default Male): ")
        gender = gender if gender.strip() in ['Male', 'Female'] else None
        
        heart_rate = input("Enter heart rate (bpm, default 80): ")
        heart_rate = float(heart_rate) if heart_rate.strip() else None
        
        body_temp = input("Enter body temperature (C, default 37): ")
        body_temp = float(body_temp) if body_temp.strip() else None
        
        oxygen_sat = input("Enter oxygen saturation (%, default 95): ")
        oxygen_sat = float(oxygen_sat) if oxygen_sat.strip() else None
        
        blood_pressure = input("Enter blood pressure (e.g., 120/80, default 120/80): ")
        blood_pressure = blood_pressure if blood_pressure.strip() else None
        
        result = parse_symptoms(
            text,
            age=age,
            gender=gender,
            heart_rate=heart_rate,
            body_temp=body_temp,
            oxygen_sat=oxygen_sat,
            blood_pressure=blood_pressure
        )
        
        print("\nInput:", result["input_text"])
        print("Extracted Symptoms:", result["extracted_symptoms"])
        print("Potential Diseases with Probabilities:")
        for disease, prob in result["potential_diseases"].items():
            print(f"{disease}: {prob:.2f}")
        print()