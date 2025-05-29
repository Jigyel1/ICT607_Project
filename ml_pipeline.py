import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from uuid import uuid4
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
import logging

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# Custom wrapper for Keras model
class KerasClassifierWrapper:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def fit(self, X, y, **kwargs):
        return self.keras_model.fit(X, y, **kwargs)

    def predict(self, X):
        return (self.keras_model.predict(X, verbose=0) > 0.5).astype("int32")

    def predict_proba(self, X):
        return self.keras_model.predict(X, verbose=0)

# Define the machine learning pipeline
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
        self.ada_model = None
        self.rf_cs_model = None
        self.lr_model = None
        self.history = None
        self.training_times = {}
        self.label_encoders = {}

    def load_dataset(self):
        """Load dataset from CSV file and check for duplicates."""
        self.df = pd.read_csv(self.filepath)
        # Check for duplicates
        logging.info(f"Duplicate rows in dataset: {self.df.duplicated().sum()}")
        print(f"Duplicate rows in dataset: {self.df.duplicated().sum()}")
        self.df = self.df.drop_duplicates()
        # Check class distribution
        logging.info(f"Class distribution:\n{self.df['Label'].value_counts(normalize=True)}")
        print("Class distribution:\n", self.df["Label"].value_counts(normalize=True))
        # Log dataset columns
        logging.info(f"Dataset columns: {list(self.df.columns)}")
        print("Dataset columns:", list(self.df.columns))

    def preprocess_data(self):
        """Encode categorical variables and scale features, avoiding data leakage."""
        # Drop potential leakage features
        leakage_columns = ['attack_cat', 'Attack']
        for col in leakage_columns:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)
                logging.info(f"Dropped '{col}' column to prevent leakage")
                print(f"Dropped '{col}' column to prevent leakage")

        X = self.df.drop("Label", axis=1)
        y = self.df["Label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Encode categorical variables on training data only
        for col in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            self.label_encoders[col] = le
            # Handle unseen categories in test set
            X_test[col] = X_test[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
            le.classes_ = np.append(le.classes_, '<unknown>')
            X_test[col] = le.transform(X_test[col])
        
        # Apply feature selection to reduce highly correlated features
        selector = SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))
        self.X_train_scaled = selector.fit_transform(X_train, y_train)
        self.X_test_scaled = selector.transform(X_test)
        selected_features = X_train.columns[selector.get_support()].tolist()
        logging.info(f"Selected features: {selected_features}")
        print("Selected features:", selected_features)
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train_scaled)
        self.X_test_scaled = scaler.transform(self.X_test_scaled)
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # Split training data for validation
        self.X_train_sub, self.X_val, self.y_train_sub, self.y_val = train_test_split(
            self.X_train_scaled, self.y_train, test_size=0.1, random_state=42, stratify=y_train)
        
        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        self.X_train_scaled, self.y_train = smote.fit_resample(self.X_train_scaled, self.y_train)
        logging.info(f"Post-SMOTE class distribution:\n{pd.Series(self.y_train).value_counts(normalize=True)}")
        print("Post-SMOTE class distribution:\n", pd.Series(self.y_train).value_counts(normalize=True))

    def check_data_leakage(self):
        """Check for feature-target correlations and low-variance features."""
        # Use only numeric columns for correlation
        numeric_df = self.df.select_dtypes(include=[np.number])
        if 'Label' not in numeric_df.columns:
            logging.warning("Label column is not numeric or missing from numeric_df")
            print("Warning: Label column is not numeric or missing from numeric_df")
            return
        correlation_matrix = numeric_df.corr()
        correlations = correlation_matrix['Label'].drop('Label', errors='ignore').sort_values(ascending=False)
        logging.info(f"Top 10 features correlated with Label:\n{correlations.head(10)}")
        print("\nTop 10 Features Most Correlated with Label:\n", correlations.head(10))
        # Flag highly correlated features
        high_corr = correlations[abs(correlations) > 0.9]
        if not high_corr.empty:
            logging.warning(f"Highly correlated features (>|0.9|): {high_corr.index.tolist()}")
            print("Warning: Highly correlated features (>|0.9|):", high_corr.index.tolist())
        
        # Check feature variances
        variances = numeric_df.drop("Label", axis=1, errors='ignore').var()
        low_variance = variances[variances < 0.01]
        if not low_variance.empty:
            logging.info(f"Low-variance features: {low_variance.index.tolist()}")
            print("Low-variance features:", low_variance.index.tolist())


    def cross_validate_model(self, model, model_name):
        """Perform stratified k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(self.X_train_scaled, self.y_train):
            X_tr, X_val = self.X_train_scaled[train_idx], self.X_train_scaled[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            model.fit(X_tr, y_tr)
            scores.append(accuracy_score(y_val, model.predict(X_val)))
        logging.info(f"{model_name} CV scores: {scores}, Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        print(f"{model_name} CV scores: {scores}, Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        return np.mean(scores)

    def train_random_forest(self):
        """Train Random Forest with regularization and class weights."""
        start = time.time()
        self.rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10,
                                        class_weight='balanced', random_state=42, n_jobs=-1)
        self.rf.fit(self.X_train_scaled, self.y_train)
        end = time.time()
        self.training_times['Random Forest'] = end - start
        preds = self.rf.predict(self.X_test_scaled)
        print("Random Forest Classification Report:\n", classification_report(self.y_test, preds))
        self.cross_validate_model(self.rf, "Random Forest")

    def train_xgboost(self):
        """Train XGBoost with regularization and class weights."""
        start = time.time()
        scale_pos_weight = sum(self.y_train == 0) / sum(self.y_train == 1)
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=5,
                                           reg_lambda=1, scale_pos_weight=scale_pos_weight,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.xgb_model.fit(self.X_train_scaled, self.y_train)
        end = time.time()
        self.training_times['XGBoost'] = end - start
        preds = self.xgb_model.predict(self.X_test_scaled)
        print("XGBoost Classification Report:\n", classification_report(self.y_test, preds))
        self.cross_validate_model(self.xgb_model, "XGBoost")

    def train_dnn(self):
        """Train Deep Neural Network with regularization and class weights."""
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(enumerate(class_weights))
        self.dnn_model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],),
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        self.dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        start = time.time()
        self.history = self.dnn_model.fit(self.X_train_scaled, self.y_train, epochs=10,
                                         batch_size=64, validation_split=0.1, class_weight=class_weight_dict, verbose=1)
        end = time.time()
        self.training_times['DNN'] = end - start
        preds = (self.dnn_model.predict(self.X_test_scaled, verbose=0) > 0.5).astype("int32")
        print("DNN Classification Report:\n", classification_report(self.y_test, preds))
        # Cross-validate DNN
        wrapped_model = KerasClassifierWrapper(self.dnn_model)
        self.cross_validate_model(wrapped_model, "DNN")

    def train_adaboost(self):
        """Train AdaBoost classifier."""
        start = time.time()
        self.ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
        self.ada_model.fit(self.X_train_scaled, self.y_train)
        end = time.time()
        self.training_times['AdaBoost'] = end - start
        self.ada_preds = self.ada_model.predict(self.X_test_scaled)
        print("AdaBoost Report:\n", classification_report(self.y_test, self.ada_preds))
        self.cross_validate_model(self.ada_model, "AdaBoost")

    def train_cost_sensitive_rf(self):
        """Train Cost-Sensitive Random Forest."""
        start = time.time()
        self.rf_cs_model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10,
                                                 class_weight='balanced', random_state=42, n_jobs=-1)
        self.rf_cs_model.fit(self.X_train_scaled, self.y_train)
        end = time.time()
        self.training_times['Cost-Sensitive RF'] = end - start
        self.rf_cs_preds = self.rf_cs_model.predict(self.X_test_scaled)
        print("Cost-Sensitive RF Report:\n", classification_report(self.y_test, self.rf_cs_preds))
        self.cross_validate_model(self.rf_cs_model, "Cost-Sensitive RF")

    def compute_subset_metrics(self, model, model_name):
        """Compute training and validation accuracy over data subsets."""
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        train_acc = []
        val_acc = []
        n_samples = len(self.X_train_sub)
        for frac in fractions:
            subset_size = int(n_samples * frac)
            X_subset = self.X_train_sub[:subset_size]
            y_subset = self.y_train_sub[:subset_size]
            model.fit(X_subset, y_subset)
            train_acc.append(accuracy_score(y_subset, model.predict(X_subset)))
            val_acc.append(accuracy_score(self.y_val, model.predict(self.X_val)))
        return train_acc, val_acc

    # Visualization methods (unchanged for brevity, but ensured to work with updates)
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
        plt.barh(range(len(indices)), importances[indices], color='skyblue')
        plt.yticks(range(len(indices)), [self.X_test.columns[i] for i in indices])
        plt.title("Random Forest: Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("rf_feature_importance.png")
        plt.close()

    def visualize_xgb_importance(self):
        """Visualize feature importances from XGBoost."""
        xgb.plot_importance(self.xgb_model, max_num_features=10, importance_type='gain', height=0.5)
        plt.title("XGBoost: Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("xgb_feature_importance.png")
        plt.close()

    def visualize_adaboost_importance(self):
        """Visualize feature importances from AdaBoost."""
        importances = self.ada_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.barh(range(len(indices)), importances[indices], color='orange')
        plt.yticks(range(len(indices)), [self.X_test.columns[i] for i in indices])
        plt.title("AdaBoost: Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("ada_feature_importance.png")
        plt.close()

    def visualize_cost_sensitive_rf_importance(self):
        """Visualize feature importances from Cost-Sensitive RF."""
        importances = self.rf_cs_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.barh(range(len(indices)), importances[indices], color='green')
        plt.yticks(range(len(indices)), [self.X_test.columns[i] for i in indices])
        plt.title("Cost-Sensitive RF: Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("cost_sensitive_rf_feature_importance.png")
        plt.close()

    def visualize_dnn_permutation_importance(self):
        """Estimate DNN feature importance using permutation importance."""
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.X_test_scaled), size=min(10000, len(self.X_test_scaled)), replace=False)
        X_test_sample = self.X_test_scaled[sample_indices]
        y_test_sample = self.y_test.iloc[sample_indices] if isinstance(self.y_test, pd.Series) else self.y_test[sample_indices]
        wrapped_model = KerasClassifierWrapper(self.dnn_model)
        result = permutation_importance(
            estimator=wrapped_model, X=X_test_sample, y=y_test_sample,
            n_repeats=2, random_state=42, scoring='accuracy'
        )
        importances = result.importances_mean
        indices = np.argsort(importances)[-10:]
        plt.barh(range(len(indices)), importances[indices], color='red')
        plt.yticks(range(len(indices)), [self.X_test.columns[i] for i in indices])
        plt.title("DNN: Top 10 Permutation Importances")
        plt.tight_layout()
        plt.savefig("dnn_feature_importance.png")
        plt.close()

    def visualize_dnn_metrics(self):
        """Plot DNN accuracy during training."""
        plt.plot(self.history.history['accuracy'], label='train')
        plt.plot(self.history.history['val_accuracy'], label='val')
        plt.title("DNN Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("dnn_accuracy.png")
        plt.close()

    def visualize_rf_metrics(self):
        """Plot Random Forest accuracy over data subsets."""
        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
        train_acc, val_acc = self.compute_subset_metrics(model, "Random Forest")
        plt.plot([20, 40, 60, 80, 100], train_acc, label='train')
        plt.plot([20, 40, 60, 80, 100], val_acc, label='val')
        plt.title("Random Forest Accuracy")
        plt.xlabel("Training Data Used (%)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("rf_accuracy.png")
        plt.close()

    def visualize_xgb_metrics(self):
        """Plot XGBoost accuracy over data subsets."""
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=5,
                                  scale_pos_weight=sum(self.y_train == 0) / sum(self.y_train == 1),
                                  subsample=0.8, colsample_bytree=0.8, random_state=42)
        train_acc, val_acc = self.compute_subset_metrics(model, "XGBoost")
        plt.plot([20, 40, 60, 80, 100], train_acc, label='train')
        plt.plot([20, 40, 60, 80, 100], val_acc, label='val')
        plt.title("XGBoost Accuracy")
        plt.xlabel("Training Data Used (%)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("xgb_accuracy.png")
        plt.close()

    def visualize_adaboost_metrics(self):
        """Plot AdaBoost accuracy over data subsets."""
        model = AdaBoostClassifier(n_estimators=50, random_state=42)
        train_acc, val_acc = self.compute_subset_metrics(model, "AdaBoost")
        plt.plot([20, 40, 60, 80, 100], train_acc, label='train')
        plt.plot([20, 40, 60, 80, 100], val_acc, label='val')
        plt.title("AdaBoost Accuracy")
        plt.xlabel("Training Data Used (%)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("ada_accuracy.png")
        plt.close()

    def visualize_cost_sensitive_rf_metrics(self):
        """Plot Cost-Sensitive Random Forest accuracy over data subsets."""
        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
        train_acc, val_acc = self.compute_subset_metrics(model, "Cost-Sensitive RF")
        plt.plot([20, 40, 60, 80, 100], train_acc, label='train')
        plt.plot([20, 40, 60, 80, 100], val_acc, label='val')
        plt.title("Cost-Sensitive Random Forest Accuracy")
        plt.xlabel("Training Data Used (%)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("rf_cs_accuracy.png")
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

        dnn_preds = (self.dnn_model.predict(self.X_test_scaled, verbose=0) > 0.5).astype("int32")
        cm = confusion_matrix(self.y_test, dnn_preds)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("DNN Confusion Matrix")
        plt.savefig("dnn_confusion_matrix.png")
        plt.close()

        cm_ada = confusion_matrix(self.y_test, self.ada_preds)
        sns.heatmap(cm_ada, annot=True, fmt="d", cmap="Oranges")
        plt.title("AdaBoost Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("adaboost_confusion_matrix.png")
        plt.close()

        cm_cs = confusion_matrix(self.y_test, self.rf_cs_preds)
        sns.heatmap(cm_cs, annot=True, fmt="d", cmap="Greens")
        plt.title("Cost-Sensitive RF Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("cost_sensitive_rf_confusion_matrix.png")
        plt.close()

        ConfusionMatrixDisplay.from_estimator(self.lr_model, self.X_test_scaled, self.y_test)
        plt.title("Logistic Regression Confusion Matrix")
        plt.savefig("lr_confusion_matrix.png")
        plt.close()

    def visualize_roc_curves(self):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.rf.predict_proba(self.X_test_scaled)[:, 1])
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(self.y_test, self.rf.predict_proba(self.X_test_scaled)[:, 1]):.2f})')

        fpr_xgb, tpr_xgb, _ = roc_curve(self.y_test, self.xgb_model.predict_proba(self.X_test_scaled)[:, 1])
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(self.y_test, self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]):.2f})')

        dnn_probs = self.dnn_model.predict(self.X_test_scaled, verbose=0).ravel()
        fpr_dnn, tpr_dnn, _ = roc_curve(self.y_test, dnn_probs)
        plt.plot(fpr_dnn, tpr_dnn, label=f'DNN (AUC = {roc_auc_score(self.y_test, dnn_probs):.2f})')

        fpr_ada, tpr_ada, _ = roc_curve(self.y_test, self.ada_model.predict_proba(self.X_test_scaled)[:, 1])
        plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {roc_auc_score(self.y_test, self.ada_model.predict_proba(self.X_test_scaled)[:, 1]):.2f})')

        fpr_rfcs, tpr_rfcs, _ = roc_curve(self.y_test, self.rf_cs_model.predict_proba(self.X_test_scaled)[:, 1])
        plt.plot(fpr_rfcs, tpr_rfcs, label=f'Cost-Sensitive RF (AUC = {roc_auc_score(self.y_test, self.rf_cs_model.predict_proba(self.X_test_scaled)[:, 1]):.2f})')

        fpr_lr, tpr_lr, _ = roc_curve(self.y_test, self.lr_model.predict_proba(self.X_test_scaled)[:, 1])
        plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(self.y_test, self.lr_model.predict_proba(self.X_test_scaled)[:, 1]):.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.savefig("roc_comparison_plot.png")
        plt.close()

    def visualize_training_time(self):
        """Compare training times for each model."""
        plt.bar(self.training_times.keys(), self.training_times.values())
        plt.title("Model Training Time")
        plt.ylabel("Seconds")
        plt.savefig("training_time.png")
        plt.close()

    def plot_model_comparison(self):
        """Compute and display model performance metrics in a grouped bar chart."""
        metrics = {
            "Random Forest": self.rf.predict(self.X_test_scaled),
            "XGBoost": self.xgb_model.predict(self.X_test_scaled),
            "DNN": (self.dnn_model.predict(self.X_test_scaled, verbose=0) > 0.5).astype("int32").flatten(),
            "AdaBoost": self.ada_preds,
            "Cost-Sensitive RF": self.rf_cs_preds
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
        df_results.plot(kind="bar", figsize=(12, 6))
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("model_comparison_plot.png")
        plt.close()

if __name__ == "__main__":
    pipeline = MLPipeline("NF-UNSW-NB15-V2.csv")
    pipeline.load_dataset()
    pipeline.preprocess_data()
    pipeline.check_data_leakage()
    pipeline.train_random_forest()
    pipeline.train_xgboost()
    pipeline.train_dnn()
    pipeline.train_adaboost()
    pipeline.train_cost_sensitive_rf()
    # pipeline.visualize_heatmap()
    # pipeline.visualize_label_distribution()
    # pipeline.visualize_normalized_labels()
    # pipeline.visualize_rf_importance()
    # pipeline.visualize_xgb_importance()
    # pipeline.visualize_adaboost_importance()
    # pipeline.visualize_cost_sensitive_rf_importance()
    # pipeline.visualize_dnn_permutation_importance()
    # pipeline.visualize_dnn_metrics()
    # pipeline.visualize_rf_metrics()
    # pipeline.visualize_xgb_metrics()
    # pipeline.visualize_adaboost_metrics()
    # pipeline.visualize_cost_sensitive_rf_metrics()
    # pipeline.visualize_pca_projection()
    # pipeline.visualize_confusion_matrices()
    # pipeline.visualize_roc_curves()
    # pipeline.visualize_training_time()
    pipeline.plot_model_comparison()