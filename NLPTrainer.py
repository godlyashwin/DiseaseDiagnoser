import subprocess as cmd
import sys
from typing import Tuple, List, Dict, Optional
from importlib import metadata
try:
    print("Importing pandas...")
    import pandas as pd
    print("Importing numpy...")
    import numpy as np
    print("Importing tensorflow...")
    import tensorflow as tf
    print("Importing sklearn.preproccesing...")
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    print("Importing sklearn.compose...")
    from sklearn.compose import ColumnTransformer
    print("Importing sklearn.pipeline...")
    from sklearn.pipeline import Pipeline
    print("Importing sklearn.feature_extraction.text...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("Importing spacy")
    import spacy
    print("Successfully installed all modules!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed correctly.")
    sys.exit(1)
def runCMD(code):
    parsedCode = code.split()
    try:
        result = cmd.run(parsedCode, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except cmd.CalledProcessError as e:
        print(f"Command failed. Return code: {e.returncode}")
        print(f"Error: {e.stderr}")
        return False
if __name__ == "__main__":
    print(" ")
    print("Python executable:", sys.executable)

    # Check and install required packages with specific versions
    required_packages = {
        "numpy": "1.23.5",
        "pandas": "2.0.3",
        "tensorflow": "2.12.0",
        "scikit_learn": "1.2.2",
        "spacy": "3.5.0",
        "kaggle": "1.7.4.5",
    }
    print("Understand that scikit-learn is treated the same as sklearn")
    for pkg, version in required_packages.items():
        try:
            installed_version = metadata.version(pkg)
            if installed_version != version:
                print(f"{pkg} version {installed_version} found, but version {version} is required. Reinstalling...")
                runCMD(f"{sys.executable} -m pip install {pkg}=={version}")
            else:
                print(f"{pkg}=={version} is installed")
        except metadata.PackageNotFoundError:
            print(f"{pkg} not found, installing version {version}...")
            runCMD(f"{sys.executable} -m pip install {pkg}=={version}")
        
# Install spaCy model
try:
    spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' is installed")
except:
    print("Installing spaCy model 'en_core_web_sm'...")
    runCMD(f"{sys.executable} -m spacy download en_core_web_sm")
    print("Successfully installed spaCy model 'en_core_web_sm'")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocess dataset
def load_dataset(csv_path: str = "disease_diagnosis.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        required_columns = [
            "Patient_ID", "Age", "Gender", "Symptom_1", "Symptom_2", "Symptom_3",
            "Heart_Rate_bpm", "Body_Temperature_C", "Blood_Pressure_mmHg",
            "Oxygen_Saturation_%", "Diagnosis", "Severity", "Treatment_Plan"
        ]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise KeyError(f"Missing columns: {missing}")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

# Parse blood pressure into systolic and diastolic
def parse_blood_pressure(bp: str) -> Tuple[float, float]:
    try:
        systolic, diastolic = map(float, bp.split('/'))
        return systolic, diastolic
    except ValueError:
        return np.nan, np.nan

# Preprocess dataset
def preprocess_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Pipeline, LabelEncoder, List[str]]:
    # Combine symptoms into a single feature for each row
    df['Symptoms'] = df[['Symptom_1', 'Symptom_2', 'Symptom_3']].fillna('None').agg(' '.join, axis=1)
    
    # Split blood pressure
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure_mmHg'].apply(parse_blood_pressure).apply(pd.Series)
    
    # Collect all unique symptoms
    all_symptoms = set()
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3']:
        all_symptoms.update(df[col].dropna().str.strip().str.lower())
    all_symptoms.discard('none')
    all_symptoms = list(all_symptoms)
    
    # Features and target
    feature_columns = [
        'Age', 'Gender', 'Symptoms', 'Heart_Rate_bpm', 'Body_Temperature_C',
        'Oxygen_Saturation_%', 'Systolic_BP', 'Diastolic_BP'
    ]
    X = df[feature_columns]
    y = df['Diagnosis']
    
    # Define preprocessing pipeline
    numerical_features = ['Age', 'Heart_Rate_bpm', 'Body_Temperature_C', 'Oxygen_Saturation_%', 'Systolic_BP', 'Diastolic_BP']
    categorical_features = ['Gender']
    text_features = ['Symptoms']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('text', TfidfVectorizer(vocabulary=all_symptoms), 'Symptoms')
        ])
    
    # Fit and transform features
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_transformed = pipeline.fit_transform(X)
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = tf.keras.utils.to_categorical(y_encoded)
    
    return X_transformed, y_one_hot, pipeline, label_encoder, all_symptoms

# Build and train TensorFlow model
def train_model(X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(y.shape[1], activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(X, y, epochs=100, batch_size=16, validation_split=0.2, verbose=0)
        return model
    except Exception as e:
        print(f"Error in model creation: {e}")
        sys.exit(1)

# Extract symptoms from text using spaCy
def extract_symptoms(text: str, known_symptoms: List[str]) -> List[str]:
    doc = nlp(text.lower())
    symptoms = []
    
    # Direct matching
    for symptom in known_symptoms:
        if symptom.lower() in text.lower():
            symptoms.append(symptom)
    
    # Token-based matching
    for token in doc:
        if token.text.lower() in [s.lower() for s in known_symptoms]:
            symptoms.append(token.text)
    
    return list(set(symptoms))

# Predict potential diseases
def predict_diseases(
    symptoms: List[str],
    model: tf.keras.Model,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    all_symptoms: List[str],
    age: float = 30.0,
    gender: str = 'Male',
    heart_rate: float = 80.0,
    body_temp: float = 37.0,
    oxygen_sat: float = 95.0,
    systolic_bp: float = 120.0,
    diastolic_bp: float = 80.0
) -> Dict[str, float]:
    if not symptoms:
        return {}
    
    # Combine symptoms into a single string
    symptoms_text = ' '.join(symptoms)
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Symptoms': [symptoms_text],
        'Heart_Rate_bpm': [heart_rate],
        'Body_Temperature_C': [body_temp],
        'Oxygen_Saturation_%': [oxygen_sat],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp]
    })
    
    # Transform input
    X_input = pipeline.transform(input_data)
    
    # Predict
    probs = model.predict(X_input, verbose=0)[0]
    disease_probs = {label_encoder.classes_[i]: float(prob) for i, prob in enumerate(probs) if prob > 0.1}
    
    return disease_probs

# Main function to parse symptoms and suggest diseases
def parse_symptoms(
    text: str,
    csv_path: str = "disease_diagnosis.csv",
    age: Optional[float] = None,
    gender: Optional[str] = None,
    heart_rate: Optional[float] = None,
    body_temp: Optional[float] = None,
    oxygen_sat: Optional[float] = None,
    blood_pressure: Optional[str] = None
) -> Dict:
    df = load_dataset(csv_path)
    
    # Preprocess and train
    X, y, pipeline, label_encoder, all_symptoms = preprocess_dataset(df)
    model = train_model(X, y)
    
    # Extract symptoms from text
    symptoms = extract_symptoms(text, all_symptoms)
    
    # Parse optional inputs
    input_params = {
        'age': age if age is not None else 30.0,
        'gender': gender if gender is not None else 'Male',
        'heart_rate': heart_rate if heart_rate is not None else 80.0,
        'body_temp': body_temp if body_temp is not None else 37.0,
        'oxygen_sat': oxygen_sat if oxygen_sat is not None else 95.0,
        'systolic_bp': 120.0,
        'diastolic_bp': 80.0
    }
    
    if blood_pressure:
        systolic, diastolic = parse_blood_pressure(blood_pressure)
        if not np.isnan(systolic):
            input_params['systolic_bp'] = systolic
            input_params['diastolic_bp'] = diastolic
    
    # Predict diseases
    disease_probs = predict_diseases(
        symptoms, model, pipeline, label_encoder, all_symptoms, **input_params
    )
    
    return {
        "input_text": text,
        "extracted_symptoms": symptoms,
        "potential_diseases": disease_probs
    }