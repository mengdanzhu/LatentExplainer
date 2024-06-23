import os
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_paragraphs_from_file(file_path):
    """Read the paragraphs from the file, return a list with each paragraph as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        paragraphs = content.split('\n\n')
        paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip() != '']
    return paragraphs

def load_predictions_from_file(file_path):
    """Read the single-line prediction from the file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        prediction = file.readline().strip()
    return prediction

def load_references_and_predictions(dataset_folder, highest_similarity_folder):
    references = {}
    predictions = {}
    
    for model in ['combination/vae', 'conditional/vae', 'conditional/diffusion', 'invariance/vae', 'invariance/diffusion']:
        references[model] = {}
        predictions[model] = {}
        
        ref_model_path = os.path.join(dataset_folder, model)
        pred_model_path = os.path.join(highest_similarity_folder, model)
        
        for dataset in os.listdir(ref_model_path):
            ref_dataset_path = os.path.join(ref_model_path, dataset)
            pred_dataset_path = os.path.join(pred_model_path, dataset)
            
            references[model][dataset] = []
            predictions[model][dataset] = []
            
            for ref_file in os.listdir(ref_dataset_path):
                ref_file_path = os.path.join(ref_dataset_path, ref_file)
                references[model][dataset].append(load_paragraphs_from_file(ref_file_path))
                
                pred_file_path = os.path.join(pred_dataset_path, ref_file)
                if os.path.exists(pred_file_path):
                    predictions[model][dataset].append(load_predictions_from_file(pred_file_path))
    
    return references, predictions

import evaluate
import csv

bleu = evaluate.load("bleu")
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

def list_to_dict(descriptions):
    return [{"image_id": str(i+1), "caption": desc} for i, desc in enumerate(descriptions)]

def calculate_scores(references, predictions):
    # Format data for COCO captioning
    pred_list = {str(i+1): [pred] for i, pred in enumerate(predictions)}
    print('pred_list:    ' ,pred_list)
    ref_list = {str(i+1): [ref for ref in refs] for i, refs in enumerate(references)}
    print('ref_list:    ' ,ref_list)

    # # Tokenize captions
    # tokenizer = PTBTokenizer()
    # pred_list = tokenizer.tokenize(pred_list)
    # ref_list = tokenizer.tokenize(ref_list)

    # Compute CIDEr
    cider = Cider()
    cider_score, _ = cider.compute_score(ref_list, pred_list)
    print(f'CIDEr Score: {cider_score}')

    # Compute SPICE
    spice = Spice()
    spice_score, _ = spice.compute_score(ref_list, pred_list)
    print(f'SPICE Score: {spice_score}')

    return cider_score, spice_score

def compute_metrics(predictions, references):
    results = {}
    
    print(predictions)
    print(references)
    bleu_results = bleu.compute(predictions=predictions, references=references)
    rouge_results = rouge.compute(predictions=predictions, references=references, use_aggregator=True)
    meteor_results = meteor.compute(predictions=predictions, references=references)
    
    cider_score, spice_score = calculate_scores(references, predictions)
    
    results['bleu'] = round(bleu_results['bleu']*100, 1)
    results['rougeL'] = round(rouge_results['rougeL']*100, 1)
    results['meteor'] = round(meteor_results['meteor']*100, 1)
    results['cider'] = round(cider_score*100, 1)
    results['spice'] = round(spice_score*100, 1)
    
    return results

def save_results_to_file(output_path, results):
    with open(os.path.join(output_path, 'output_ablation_no_uq.csv'), mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Dataset', 'BLEU', 'ROUGE-L', 'METEOR', 'CIDEr', 'SPICE'])
        
        for model, datasets in results.items():
            for dataset, metrics in datasets.items():
                writer.writerow([model, dataset, metrics['bleu'], metrics['rougeL'], metrics['meteor'], metrics['cider'], metrics['spice']])


dataset_folder = '/home/Documents/experiments/annotations'
highest_similarity_folder = '/home/Documents/evaluation/result/highest_similarity_paragraphs'
output_path = '/home/Documents/evaluation/result/metrics'

references, predictions = load_references_and_predictions(dataset_folder, highest_similarity_folder)

results = {}

for model in references:
    results[model] = {}
    for dataset in references[model]:
        print(model)
        #refs = [[ref] for ref in references[model][dataset]]  # Wrap each reference in a list
        preds = predictions[model][dataset]
        refs = references[model][dataset]
        
        metrics = compute_metrics(preds, refs)
        results[model][dataset] = metrics

save_results_to_file(output_path, results)
