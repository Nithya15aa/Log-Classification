# scripts/train_ml_enhanced.py

import argparse
import pickle
import os

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.config import settings
from app.feature_extractor import LogFeatureExtractor
from app.utils import preprocess_log


def train(csv_path: str, text_col: str, label_col: str, test_csv_path: str = None, output_metrics_path: str = "evaluation_metrics.txt"):
	"""
	Enhanced training with combined features.
	Trains on the full dataset and saves the model.
	"""

	# Load input file
	sheet = pd.read_csv(csv_path)
	sheet = sheet[[text_col, label_col]].dropna()

	log_texts = sheet[text_col].tolist()
	log_tags = sheet[label_col].tolist()

	# Filter out labels with < 2 samples
	tag_counts = pd.Series(log_tags).value_counts()
	allowed_tags = tag_counts[tag_counts >= 2].index

	filtered_sheet = sheet[sheet[label_col].isin(allowed_tags)]
	log_texts = filtered_sheet[text_col].tolist()
	log_tags = filtered_sheet[label_col].tolist()

	print("Dataset:", len(log_texts), "samples across", len(set(log_tags)), "classes")

	if len(log_texts) == 0:
		print("Error: No usable training samples after filtering.")
		return

	# Step 1: Sentence embeddings
	print("Generating sentence embeddings...")
	prepared_texts = [preprocess_log(t) for t in log_texts]

	embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
	text_vectors = embedding_model.encode(prepared_texts, show_progress_bar=True)

	# Step 2: Handcrafted features
	print("Extracting domain-specific features...")
	feature_tool = LogFeatureExtractor()
	custom_features = feature_tool.extract_batch(log_texts)

	# Step 3: Combine features
	print("Combining embeddings + handcrafted features...")
	combined_features = np.hstack([text_vectors, custom_features])

	# Step 4: Encode labels
	tag_encoder = LabelEncoder()
	tag_ids = tag_encoder.fit_transform(log_tags)

	# Step 5: Class imbalance handling
	tag_weights = class_weight.compute_class_weight(
		'balanced',
		classes=np.unique(tag_ids),
		y=tag_ids
	)
	print("Class weights calculated.")

	# Step 6: Train model
	print("Training Random Forest model on full dataset...")

	classifier = RandomForestClassifier(
		n_estimators=200,
		max_depth=20,
		min_samples_split=5,
		n_jobs=-1,
		class_weight='balanced',
		random_state=42
	)

	classifier.fit(combined_features, tag_ids)
	print("Training completed!")

	# Step 7: Save model + label encoder
	os.makedirs(os.path.dirname(settings.LR_MODEL_PATH), exist_ok=True)
	os.makedirs(os.path.dirname(settings.LABEL_ENCODER_PATH), exist_ok=True)

	with open(settings.LR_MODEL_PATH, "wb") as f:
		pickle.dump(classifier, f)
	print("Model saved:", settings.LR_MODEL_PATH)

	with open(settings.LABEL_ENCODER_PATH, "wb") as f:
		pickle.dump(tag_encoder, f)
	print("Label encoder saved:", settings.LABEL_ENCODER_PATH)

	# Step 8: Optional test set evaluation
	if test_csv_path:
		if not os.path.exists(test_csv_path):
			print(f"Warning: Test CSV not found: {test_csv_path}")
			return

		print(f"\nEvaluating on test data: {test_csv_path}")
		test_sheet = pd.read_csv(test_csv_path)

		if text_col not in test_sheet.columns or label_col not in test_sheet.columns:
			print(f"Error: Test CSV must include '{text_col}' and '{label_col}' columns.")
			return

		test_sheet = test_sheet[[text_col, label_col]].dropna()
		test_texts = test_sheet[text_col].tolist()
		test_tags = test_sheet[label_col].tolist()

		if len(test_texts) == 0:
			print("Error: Test file is empty.")
			return

		# Embeddings + handcrafted features for test
		print("Generating test embeddings...")
		prepared_test_texts = [preprocess_log(t) for t in test_texts]
		test_vectors = embedding_model.encode(prepared_test_texts, show_progress_bar=True)

		print("Extracting test handcrafted features...")
		test_features = feature_tool.extract_batch(test_texts)

		combined_test_features = np.hstack([test_vectors, test_features])

		# Remove unseen labels
		known_tags = set(tag_encoder.classes_)
		valid_mask = [t in known_tags for t in test_tags]

		if not all(valid_mask):
			removed = len(test_tags) - sum(valid_mask)
			print(f"Warning: {removed} test samples dropped due to unknown labels.")
			combined_test_features = combined_test_features[valid_mask]
			test_tags = [t for t, keep in zip(test_tags, valid_mask) if keep]

		if len(test_tags) == 0:
			print("Error: No valid test samples left.")
			return

		test_tag_ids = tag_encoder.transform(test_tags)

		print("Predicting on test set...")
		test_predictions = classifier.predict(combined_test_features)

		test_accuracy = accuracy_score(test_tag_ids, test_predictions)
		test_report = classification_report(test_tag_ids, test_predictions, target_names=tag_encoder.classes_)
		test_matrix = confusion_matrix(test_tag_ids, test_predictions)

		print(f"Test Accuracy: {test_accuracy:.4f}")
		print("Classification Report:\n", test_report)

		with open(output_metrics_path, "w") as f:
			f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
			f.write("Classification Report:\n")
			f.write(test_report)
			f.write("\n\nConfusion Matrix:\n")
			f.write(str(test_matrix))

		print(f"Metrics saved to {output_metrics_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Enhanced ML training with feature engineering")
	parser.add_argument("--csv", required=True, help="Path to training CSV")
	parser.add_argument("--text-col", default="text", help="Column with log messages")
	parser.add_argument("--label-col", default="label", help="Column with labels")
	parser.add_argument("--test-csv", help="Path to test CSV")
	parser.add_argument("--output-metrics", default="evaluation_metrics.txt", help="Where to save metrics")
	args = parser.parse_args()

	train(args.csv, args.text_col, args.label_col, args.test_csv, args.output_metrics)
