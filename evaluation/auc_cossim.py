from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import torch



def load_paragraphs_from_file(file_path):
    """Read the paragraphs from the file, return a list with each paragraph as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        paragraphs = content.split('\n\n')
        paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip() != '']
    return paragraphs


def calculate_cosine_similarity(paragraphs):
    vectorizer = CountVectorizer().fit_transform(paragraphs)
    cosine_sim = cosine_similarity(vectorizer)
    return cosine_sim

def get_uncertainty_score(cosine_sim):
    n = cosine_sim.shape[0]
    sum_sim = np.sum(cosine_sim) - np.trace(cosine_sim)
    sum_sim = sum_sim 
    C = n * (n - 1) 
    avg_sim = sum_sim / C
    return avg_sim

def find_max_mean_cosine_similarity_paragraph(cosine_sim, paragraphs):
    mean_cosine_sim = np.mean(cosine_sim, axis=1)
    max_index = np.argmax(mean_cosine_sim)
    return paragraphs[max_index]

base_folders = [
    'combination/vae', 'conditional/vae', 'conditional/diffusion',
    'invariance/diffusion', 'invariance/vae'
]

output_folder = '/home/Documents/evaluation/result/highest_similarity_paragraphs'

auc_folder = '/home/Documents/evaluation/result/auc'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

most_similar_list = []
true_labels = []
uncertainty_scores = {}
filtered_most_similar_list = []
filtered_true_labels = []

no_pattern = ["x0_gen-Edit_zt-Examples_9-edit_0.7T-mid-block_0-pc_003_pos-edit_prompt_ipad.txt",
"x0_gen-Edit_zt-Examples_9-edit_0.7T-mid-block_0-pc_004_neg-edit_prompt_ipad.txt",
"x0_gen-Edit_zt-Examples_10-edit_0.7T-mid-block_0-pc_003_neg-edit_prompt_smile.txt",
"x0_gen-Edit_zt-Examples_21-edit_0.7T-mid-block_0-pc_003_pos-edit_prompt_t-shirt.txt",
"x0_gen-Edit_zt-Examples_21-edit_0.7T-mid-block_0-pc_004_pos-edit_prompt_t-shirt.txt",
"x0_gen-Edit_zt-Examples_14-edit_0.7T-mid-block_0-pc_003_pos-edit_prompt_mouth closed.txt",
"x0_gen-Edit_zt-Examples_27-edit_0.7T-mid-block_0-pc_000_neg-edit_prompt_rabbit.txt",
"x0_gen-Edit_xt-CelebA_HQ_6-edit_1.0T-mid-block_0-pc_005_pos.txt",
"x0_gen-Edit_xt-AFHQ_3-edit_1.0T-mid-block_0-pc_003_pos.txt",
"x0_gen-Edit_xt-AFHQ_7-edit_1.0T-mid-block_0-pc_004_neg.txt",
"x0_gen-Edit_xt-LSUN_church_35-edit_1.0T-mid-block_0-pc_002_neg.txt",
"x0_gen-Edit_xt-LSUN_church_82-edit_1.0T-mid-block_0-pc_004_pos.txt"]

for base_folder in base_folders:
    folder_path = os.path.join('/home/Documents/evaluation/result', base_folder)
    for item in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, item)
        if os.path.isfile(full_path) and item.lower().endswith('.txt'):
            paragraphs = load_paragraphs_from_file(full_path)
            print(paragraphs)
            cosine_sim = calculate_cosine_similarity(paragraphs)
            uncertainty_score = get_uncertainty_score(cosine_sim)
            uncertainty_scores[item] = uncertainty_score

            highest_similarity_paragraph = find_max_mean_cosine_similarity_paragraph(cosine_sim, paragraphs)
            output_file_path = os.path.join(output_folder, base_folder, item)
            if not os.path.exists(os.path.join(output_folder, base_folder)):
                os.makedirs(os.path.join(output_folder, base_folder))
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(highest_similarity_paragraph)

            most_similar_list.append(uncertainty_score)

            # Update true_labels based on your criteria
            if item in no_pattern:
                true_labels.append(0)
            else:
                true_labels.append(1)


predicted_probabilities = most_similar_list

fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities, pos_label=1)
roc_auc = roc_auc_score(true_labels, predicted_probabilities)


auc_scores = []
for threshold in thresholds:
    y_pred = (predicted_probabilities >= threshold).astype(int)
    auc = roc_auc_score(true_labels, y_pred)
    #auc = roc_auc_score(filtered_true_labels, y_pred)

    auc_scores.append(auc)

best_threshold = thresholds[np.argmax(auc_scores)]
best_auc = max(auc_scores)

binary_predictions = (np.array(predicted_probabilities) >= best_threshold).astype(int)

precision = precision_score(true_labels, binary_predictions)
recall = recall_score(true_labels, binary_predictions)
f1 = f1_score(true_labels, binary_predictions)


# Save uncertainty scores and AUC metrics to a file
uncertainty_scores_output_path = os.path.join(auc_folder, 'new_cossim.txt')

with open(uncertainty_scores_output_path, 'w', encoding='utf-8') as file:
    for item, score in uncertainty_scores.items():
        file.write(f"{item}: {score}\n")
    file.write(f"\nbest threshold: {best_threshold}\n")
    file.write(f"best auc: {best_auc}\n")
    file.write(f"precision: {precision}\n")
    file.write(f"recall: {recall}\n")
    file.write(f"F1 score: {f1}\n")

print("Best Threshold:", best_threshold)
print("Max AUC:", best_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

