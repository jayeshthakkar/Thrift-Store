from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)  # Convert scores to binary predictions
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Initialize lists to store metrics for plotting
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Collect predictions for evaluation
y_true_all = []
y_score_all = []
processed_users = 0

print("\nCollecting predictions for evaluation...")
unique_users = test_data['user_id'].unique()

for user_id in tqdm(unique_users[:1000], desc="Processing users"):
    y_true, y_scores = get_user_predictions(user_id, train_matrix, test_data, model_knn)
    if y_true is not None and len(y_true) > 0:
        y_true_all.extend(y_true)
        y_score_all.extend(y_scores)
        processed_users += 1

        # Calculate and store metrics for each user
        accuracy, precision, recall, f1 = calculate_metrics(y_true, y_scores)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

# Convert to numpy arrays
y_true_all = np.array(y_true_all)
y_score_all = np.array(y_score_all)

print(f"\nSuccessfully processed {processed_users} users")
print(f"Total predictions: {len(y_true_all)}")

# Plot ROC curve and calculate AUC
if len(y_true_all) > 0:
    print("\nPlotting ROC curve...")
    roc_auc = plot_roc_curve(y_true_all, y_score_all)
    print(f"\nOverall AUC Score: {roc_auc:.3f}")

    # Print additional metrics
    print("\nEvaluation Summary:")
    print(f"Positive ratings ratio: {np.mean(y_true_all):.2%}")
    print(f"Average prediction score: {np.mean(y_score_all):.3f}")
else:
    print("No predictions were generated for evaluation.")

# Box plot for accuracy, precision, recall, and F1 score
metrics_df = pd.DataFrame({
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores
})

# Create box plots
plt.figure(figsize=(12, 8))
sns.boxplot(data=metrics_df)
plt.title('Performance Metrics Comparison')
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.show()

# Comparison bar chart
avg_metrics = metrics_df.mean()
plt.figure(figsize=(8, 6))
avg_metrics.plot(kind='bar', color=['blue', 'green', 'red', 'orange'])
plt.title('Average Performance Metrics Comparison')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.show()

