import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class StudentPredictor:
    def __init__(self):
        self.enrollment_model = None
        self.graduation_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_sample_data(self, n_students=1000):
        """Generate realistic sample student data for demonstration"""
        np.random.seed(42)
        
        data = {
            'student_id': range(n_students),
            'high_school_gpa': np.random.normal(3.2, 0.5, n_students),
            'sat_score': np.random.normal(1200, 200, n_students),
            'family_income': np.random.normal(60000, 30000, n_students),
            'distance_from_campus': np.random.exponential(50, n_students),
            'campus_visits': np.random.poisson(2, n_students),
            'first_generation': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
            'financial_aid_amount': np.random.normal(10000, 5000, n_students),
            'program_applied': np.random.choice(['Engineering', 'Business', 'Arts', 'Sciences'], n_students),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target variables
        # Enrollment likelihood based on features
        enrollment_prob = (
            0.3 + 
            0.1 * (df['high_school_gpa'] > 3.5) +
            0.2 * (df['financial_aid_amount'] > 12000) +
            0.15 * (df['campus_visits'] > 1) -
            0.1 * (df['distance_from_campus'] > 100)
        )
        df['enrolled'] = np.random.binomial(1, np.clip(enrollment_prob, 0, 1))
        
        # Graduation likelihood for enrolled students
        graduation_prob = (
            0.4 +
            0.3 * (df['high_school_gpa'] > 3.0) +
            0.2 * (df['sat_score'] > 1100) -
            0.2 * (df['first_generation'] == 1) +
            0.1 * (df['family_income'] > 50000)
        )
        df['graduated'] = np.random.binomial(1, np.clip(graduation_prob, 0, 1))
        # Only enrolled students can graduate
        df.loc[df['enrolled'] == 0, 'graduated'] = 0
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for modeling"""
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['program_applied']
        for col in categorical_cols:
            if col in df_processed.columns:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        return df_processed
    
    def train_enrollment_model(self, df):
        """Train model to predict enrollment"""
        print("Training Enrollment Prediction Model...")
        
        # Features for enrollment prediction
        enrollment_features = [
            'high_school_gpa', 'sat_score', 'family_income', 'distance_from_campus',
            'campus_visits', 'first_generation', 'financial_aid_amount', 'program_applied'
        ]
        
        X = df[enrollment_features]
        y = df['enrolled']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.enrollment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.enrollment_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.enrollment_model.predict(X_test_scaled)
        print("Enrollment Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nEnrollment Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train, X_test, y_train, y_test
    
    def train_graduation_model(self, df):
        """Train model to predict graduation (only on enrolled students)"""
        print("\nTraining Graduation Prediction Model...")
        
        # Only use students who enrolled
        enrolled_students = df[df['enrolled'] == 1].copy()
        
        if len(enrolled_students) == 0:
            print("No enrolled students found for graduation prediction")
            return None
        
        # Features for graduation prediction
        graduation_features = [
            'high_school_gpa', 'sat_score', 'family_income', 'first_generation',
            'financial_aid_amount', 'program_applied'
        ]
        
        X = enrolled_students[graduation_features]
        y = enrolled_students['graduated']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.graduation_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.graduation_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.graduation_model.predict(X_test_scaled)
        print("Graduation Model Accuracy:", accuracy_score(y_test, y_pred))
        print("\nGraduation Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train, X_test, y_train, y_test
    
    def predict_student_success(self, student_data):
        """Predict enrollment and graduation probability for new students"""
        if self.enrollment_model is None or self.graduation_model is None:
            raise ValueError("Models must be trained first!")
        
        # Preprocess student data
        student_processed = self.preprocess_data(student_data)
        
        # Enrollment features
        enrollment_features = [
            'high_school_gpa', 'sat_score', 'family_income', 'distance_from_campus',
            'campus_visits', 'first_generation', 'financial_aid_amount', 'program_applied'
        ]
        
        X_enroll = student_processed[enrollment_features]
        X_enroll_scaled = self.scaler.transform(X_enroll)
        
        # Predict enrollment
        enrollment_proba = self.enrollment_model.predict_proba(X_enroll_scaled)[:, 1]
        enrollment_pred = self.enrollment_model.predict(X_enroll_scaled)
        
        # Predict graduation for those likely to enroll
        graduation_features = [
            'high_school_gpa', 'sat_score', 'family_income', 'first_generation',
            'financial_aid_amount', 'program_applied'
        ]
        
        X_grad = student_processed[graduation_features]
        X_grad_scaled = self.scaler.transform(X_grad)
        graduation_proba = self.graduation_model.predict_proba(X_grad_scaled)[:, 1]
        graduation_pred = self.graduation_model.predict(X_grad_scaled)
        
        # Create results dataframe
        results = student_data.copy()
        results['enrollment_probability'] = enrollment_proba
        results['predicted_enrollment'] = enrollment_pred
        results['graduation_probability'] = graduation_proba
        results['predicted_graduation'] = graduation_pred
        
        return results
    
    def plot_feature_importance(self):
        """Plot feature importance for both models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Enrollment feature importance
        if self.enrollment_model is not None:
            enrollment_features = [
                'High School GPA', 'SAT Score', 'Family Income', 'Distance from Campus',
                'Campus Visits', 'First Generation', 'Financial Aid', 'Program'
            ]
            importances = self.enrollment_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax1.barh(range(len(indices)), importances[indices], align='center')
            ax1.set_yticks(range(len(indices)))
            ax1.set_yticklabels([enrollment_features[i] for i in indices])
            ax1.set_xlabel('Feature Importance')
            ax1.set_title('Enrollment Prediction - Feature Importance')
        
        # Graduation feature importance
        if self.graduation_model is not None:
            graduation_features = [
                'High School GPA', 'SAT Score', 'Family Income', 'First Generation',
                'Financial Aid', 'Program'
            ]
            importances = self.graduation_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax2.barh(range(len(indices)), importances[indices], align='center')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([graduation_features[i] for i in indices])
            ax2.set_xlabel('Feature Importance')
            ax2.set_title('Graduation Prediction - Feature Importance')
        
        plt.tight_layout()
        plt.show()

# Demonstration of the complete system
def main():
    # Initialize the predictor
    predictor = StudentPredictor()
    
    # Generate sample data (in real scenario, you'd load from CSV/database)
    print("Generating sample student data...")
    student_data = predictor.generate_sample_data(1000)
    
    print("Sample of the generated data:")
    print(student_data.head())
    print(f"\nData shape: {student_data.shape}")
    print(f"Enrollment rate: {student_data['enrolled'].mean():.2%}")
    print(f"Graduation rate (among enrolled): {student_data[student_data['enrolled'] == 1]['graduated'].mean():.2%}")
    
    # Preprocess data
    processed_data = predictor.preprocess_data(student_data)
    
    # Train models
    predictor.train_enrollment_model(processed_data)
    predictor.train_graduation_model(processed_data)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Generate new students for prediction
    print("\n" + "="*50)
    print("PREDICTING FOR NEW STUDENTS")
    print("="*50)
    
    new_students = predictor.generate_sample_data(10)
    new_students_processed = predictor.preprocess_data(new_students)
    
    # Make predictions
    predictions = predictor.predict_student_success(new_students)
    
    print("\nPredictions for new students:")
    display_cols = ['high_school_gpa', 'sat_score', 'financial_aid_amount', 
                   'enrollment_probability', 'predicted_enrollment', 
                   'graduation_probability', 'predicted_graduation']
    print(predictions[display_cols].round(3))
    
    # Generate actionable insights
    print("\n" + "="*50)
    print("ACTIONABLE INSIGHTS")
    print("="*50)
    
    high_risk_enrollment = predictions[predictions['enrollment_probability'] < 0.4]
    high_risk_graduation = predictions[(predictions['predicted_enrollment'] == 1) & 
                                     (predictions['graduation_probability'] < 0.5)]
    
    print(f"Students needing enrollment outreach: {len(high_risk_enrollment)}")
    for _, student in high_risk_enrollment.iterrows():
        print(f"  Student ID: {student['student_id']} - Enrollment Probability: {student['enrollment_probability']:.1%}")
        print(f"  Recommended: Personal follow-up, scholarship review, campus visit invitation")
    
    print(f"\nEnrolled students needing graduation support: {len(high_risk_graduation)}")
    for _, student in high_risk_graduation.iterrows():
        print(f"  Student ID: {student['student_id']} - Graduation Probability: {student['graduation_probability']:.1%}")
        print(f"  Recommended: Academic advising, tutoring referral, mentorship program")

if __name__ == "__main__":
    main()