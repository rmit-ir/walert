
## add evaluation metrics to falcon_results and prepare it in format for end2end_eval.R file
from eval import compute_rouge_scores, compute_bleu, compute_bertscore
import pandas as pd

DATA_DIR = "/home/ubuntu/EIP/Damiano/walert/quantitative_eval"
GROUNDTRUTH = DATA_DIR + "/data/gold_summaries.csv"

falcon_results = pd.read_csv(DATA_DIR + "/data/falcon_results.csv")

ground_truth = pd.read_csv(GROUNDTRUTH)

# Initialize new columns for the metrics
falcon_results['rouge_1_f1_top3'] = 0.0
falcon_results['rouge_2_f1_top3'] = 0.0
falcon_results['rouge_l_f1_top3'] = 0.0
falcon_results['bleu_score_top3'] = 0.0
falcon_results['bert_score_f1_top3'] = 0.0

# Iterate over each row in Falcon results and compute metrics
for index, row in falcon_results.iterrows():
    question_id = row['question_id']
    
    # Get the ground truth summary for the current question
    reference = ground_truth[ground_truth['question_id'] == question_id]['summary'].iloc[0]
    candidate = row['falcon_generated_answer']
    
    # Compute ROUGE scores
    rouge_1_f1, rouge_2_f1, rouge_l_f1 = compute_rouge_scores(candidate, reference)
    
    # Compute BLEU score
    bleu_score = compute_bleu(candidate, reference)
    
    # Compute BERTScore
    _, _, bert_score_f1 = compute_bertscore(candidate, reference)
    
    # Add the scores to the DataFrame
    falcon_results.at[index, 'gold_summ'] = reference
    falcon_results.at[index, 'rouge_1_f1_top3'] = rouge_1_f1
    falcon_results.at[index, 'rouge_2_f1_top3'] = rouge_2_f1
    falcon_results.at[index, 'rouge_l_f1_top3'] = rouge_l_f1
    falcon_results.at[index, 'bleu_score_top3'] = bleu_score
    falcon_results.at[index, 'bert_score_f1_top3'] = bert_score_f1

# Save the updated DataFrame to a new CSV file
falcon_results.to_csv(DATA_DIR + "/data/falcon_results_metrics.csv", index=False)





#print( ground_truth[ground_truth['question_id'] == 'W01Q01']['summary'].iloc[0])
print(compute_rouge_scores(falcon_results[falcon_results['question_id'] == 'W01Q01']['falcon_generated_answer'][0], ground_truth[ground_truth['question_id'] == 'W01Q01']['summary'].iloc[0]))

#print(compute_bertscore(falcon_results[falcon_results['question_id'] == 'W01Q01']['falcon_generated_answer'][0], ground_truth[ground_truth['question_id'] == 'W01Q01']['summary'].iloc[0]))

#print(compute_bleu(falcon_results[falcon_results['question_id'] == 'W01Q01']['falcon_generated_answer'][0], ground_truth[ground_truth['question_id'] == 'W01Q01']['summary'].iloc[0]))
