# Import necessary libraries for data manipulation, visualization, machine learning, and deep learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # For model explainability using SHAP values
import time  # To measure training durations

# Scikit-learn modules for preprocessing, modeling, evaluation, and dimensionality reduction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

# XGBoost for gradient boosting
import xgboost as xgb

# TensorFlow and Keras for deep neural network modeling
import tensorflow as tf
from tensorflow.keras import layers, models


# Define the machine learning pipeline as a class
class MLPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf = None
        self.xgb_model = None
        self.dnn_model = None
        self.history = None
        self.training_times = {}

    # Data load and process

    def load_dataset(self):
        """Load dataset from CSV file."""
        self.df = pd.read_csv(self.filepath)

    def preprocess_data(self):
        """Encode categorical variables and scale features."""
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = LabelEncoder().fit_transform(self.df[col])
        X = self.df.drop("Label", axis=1)
        y = self.df["Label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Model Training

    def train_random_forest(self):
        """Train Random Forest model and print classification report."""
        start = time.time()
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.rf.fit(self.X_train_scaled, self.y_train)
        end = time.time()
        self.training_times['Random Forest'] = end - start
        preds = self.rf.predict(self.X_test_scaled)
        print("Random Forest Classification Report:\n", classification_report(self.y_test, preds))

    def train_xgboost(self):
        """Train XGBoost model and print classification report."""
        start = time.time()
        self.xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.xgb_model.fit(self.X_train_scaled, self.y_train)
        end = time.time()
        self.training_times['XGBoost'] = end - start
        preds = self.xgb_model.predict(self.X_test_scaled)
        print("XGBoost Classification Report:\n", classification_report(self.y_test, preds))

    def train_dnn(self):
        """Train Deep Neural Network and print classification report."""
        self.dnn_model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        start = time.time()
        self.history = self.dnn_model.fit(self.X_train_scaled, self.y_train, epochs=10,
                                          batch_size=64, validation_split=0.2, verbose=1)
        end = time.time()
        self.training_times['DNN'] = end - start
        preds = (self.dnn_model.predict(self.X_test_scaled) > 0.5).astype("int32")
        print("DNN Classification Report:\n", classification_report(self.y_test, preds))

    def train_adaboost(self):
        """
        Train an AdaBoost classifier.
        """
        self.ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
        self.ada_model.fit(self.X_train_scaled, self.y_train)
        self.ada_preds = self.ada_model.predict(self.X_test_scaled)
        print("AdaBoost Report:\n", classification_report(self.y_test, self.ada_preds))

    def train_cost_sensitive_rf(self):
        """
        Train a cost-sensitive Random Forest using class weights.
        """
        self.rf_cs_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                                  class_weight='balanced', random_state=42)
        self.rf_cs_model.fit(self.X_train_scaled, self.y_train)
        self.rf_cs_preds = self.rf_cs_model.predict(self.X_test_scaled)
        print("Cost-Sensitive RF Report:\n", classification_report(self.y_test, self.rf_cs_preds))

    # Visualizations

    def visualize_heatmap(self):
        """Generate and save correlation heatmap."""
        sns.heatmap(self.df.corr(), cmap='coolwarm')
        plt.title("Feature Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()

    def visualize_label_distribution(self):
        """Visualize original label distribution."""
        sns.countplot(x='Label', data=self.df)
        plt.title("Label Distribution")
        plt.savefig("label_distribution.png")
        plt.close()

    def visualize_normalized_labels(self):
        """Display normalized label distribution."""
        unique, counts = np.unique(self.df["Label"], return_counts=True)
        plt.bar(unique, counts)
        plt.title("Normalized Label Distribution")
        plt.savefig("normalized_label_distribution.png")
        plt.close()

    def visualize_rf_importance(self):
        """Visualize feature importances from Random Forest."""
        importances = self.rf.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [self.X_test.columns[i] for i in indices])
        plt.title("RF Top 10 Feature Importances")
        plt.savefig("rf_feature_importance.png")
        plt.close()

    def visualize_xgb_importance(self):
        """Visualize feature importances from XGBoost."""
        xgb.plot_importance(self.xgb_model, max_num_features=10, importance_type='gain', height=0.5)
        plt.title("XGBoost Feature Importance")
        plt.savefig("xgb_feature_importance.png")
        plt.close()

    def visualize_confusion_matrices(self):
        """Generate and save confusion matrices for all models."""
        ConfusionMatrixDisplay.from_estimator(self.rf, self.X_test_scaled, self.y_test)
        plt.title("Random Forest Confusion Matrix")
        plt.savefig("rf_confusion_matrix.png")
        plt.close()

        ConfusionMatrixDisplay.from_estimator(self.xgb_model, self.X_test_scaled, self.y_test)
        plt.title("XGBoost Confusion Matrix")
        plt.savefig("xgb_confusion_matrix.png")
        plt.close()

        dnn_preds = (self.dnn_model.predict(self.X_test_scaled) > 0.5).astype("int32")
        cm = confusion_matrix(self.y_test, dnn_preds)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("DNN Confusion Matrix")
        plt.savefig("dnn_confusion_matrix.png")
        plt.close()

    def visualize_dnn_metrics(self):
        """Plot DNN accuracy and loss during training."""
        plt.plot(self.history.history['accuracy'], label='train')
        plt.plot(self.history.history['val_accuracy'], label='val')
        plt.title("DNN Accuracy")
        plt.legend()
        plt.savefig("dnn_accuracy.png")
        plt.close()

        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='val')
        plt.title("DNN Loss")
        plt.legend()
        plt.savefig("dnn_loss.png")
        plt.close()

    def visualize_roc_curves(self):
        """Plot ROC curves for all models."""
        rf_probs = self.rf.predict_proba(self.X_test_scaled)[:, 1]
        xgb_probs = self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]
        dnn_probs = self.dnn_model.predict(self.X_test_scaled).ravel()
        rf_fpr, rf_tpr, _ = roc_curve(self.y_test, rf_probs)
        xgb_fpr, xgb_tpr, _ = roc_curve(self.y_test, xgb_probs)
        dnn_fpr, dnn_tpr, _ = roc_curve(self.y_test, dnn_probs)

        plt.plot(rf_fpr, rf_tpr, label="RF")
        plt.plot(xgb_fpr, xgb_tpr, label="XGB")
        plt.plot(dnn_fpr, dnn_tpr, label="DNN")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curves")
        plt.legend()
        plt.savefig("roc_curves.png")
        plt.close()

    def visualize_pca_projection(self):
        """Visualize PCA projection of test data."""
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.X_test_scaled)
        plt.scatter(components[:, 0], components[:, 1], c=self.y_test, cmap='coolwarm', alpha=0.6)
        plt.title("PCA Projection of Test Data")
        plt.colorbar()
        plt.savefig("pca_projection.png")
        plt.close()

    def visualize_training_time(self):
        """Compare training times for each model."""
        plt.bar(self.training_times.keys(), self.training_times.values())
        plt.title("Model Training Time")
        plt.ylabel("Seconds")
        plt.savefig("training_time.png")
        plt.close()

    def visualize_shap_summary(self):
        """Visualize SHAP values for XGBoost model to interpret feature importance and contribution."""
        explainer = shap.Explainer(self.xgb_model)
        shap_values = explainer(self.X_test_scaled)
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title("SHAP Summary Plot for XGBoost")
        plt.savefig("shap_summary_plot.png")
        plt.close()
    
    
    def plot_model_comparison(self):
        """
        Compute accuracy, precision, recall, and F1-score for each model,
        and display them in a grouped bar chart.
        """
        metrics = {
            "Random Forest": self.rf.predict(self.X_test_scaled),
            "XGBoost": self.xgb_model.predict(self.X_test_scaled),
            "DNN": (self.dnn_model.predict(self.X_test_scaled) > 0.5).astype("int32").flatten(),
            "AdaBoost": self.ada_preds,
            "Cost-Sensitive RF": self.rf_cs_preds,
        }

        results = {
            "Model": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1-score": []
        }

        for model_name, preds in metrics.items():
            results["Model"].append(model_name)
            results["Accuracy"].append(accuracy_score(self.y_test, preds))
            results["Precision"].append(precision_score(self.y_test, preds))
            results["Recall"].append(recall_score(self.y_test, preds))
            results["F1-score"].append(f1_score(self.y_test, preds))

        df_results = pd.DataFrame(results)
        df_results.set_index("Model", inplace=True)
        
        # Plot grouped bar chart
        df_results.plot(kind="bar", figsize=(12, 6))
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("model_comparison_plot.png")
        plt.close()
   

# Main block to run the full ML pipeline when the script is executed
if __name__ == "__main__":
    pipeline = MLPipeline("NF-UNSW-NB15-V2.csv")  # Instantiate the pipeline with dataset path
    pipeline.load_dataset()                       # Load the dataset into a DataFrame
    pipeline.preprocess_data()                    # Encode, split, and normalize data
    pipeline.train_random_forest()                # Train and evaluate Random Forest model
    pipeline.train_xgboost()                      # Train and evaluate XGBoost model
    pipeline.train_dnn()                          # Train and evaluate Deep Neural Network
    pipeline.train_adaboost()                     # Train and evaluate Adaboost
    pipeline.train_cost_sensitive_rf()            # Train and evaluate Cost Sensitive               
    pipeline.visualize_heatmap()                  # Create correlation heatmap of features
    pipeline.visualize_label_distribution()       # Visualize original label distribution
    pipeline.visualize_normalized_labels()        # Show distribution of labels after encoding
    pipeline.visualize_rf_importance()            # Feature importance for Random Forest
    pipeline.visualize_xgb_importance()           # Feature importance for XGBoost
    pipeline.visualize_confusion_matrices()       # Show confusion matrices for all models
    pipeline.visualize_dnn_metrics()              # Plot DNN training accuracy and loss
    pipeline.visualize_roc_curves()               # Plot ROC curves for all models
    pipeline.visualize_pca_projection()           # Visualize test data in 2D using PCA
    pipeline.visualize_training_time()            # Compare training time for each model
    pipeline.visualize_shap_summary()             # Visualize SHAP summary plot for XGBoost
    pipeline.plot_model_comparison()              # Visualization compared by the accuracy, precision, recall, & f1 score of all models
