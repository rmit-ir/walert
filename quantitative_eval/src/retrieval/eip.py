
import threading
import numpy as np
import os
from pyserini.search import FaissSearcher
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time
import os
import logging

torch.cuda.empty_cache()

def get_answer(text):
    # Find the index of 'Answer:'
    answer = ''
    index = text.find('Answer:')
    if index != -1:
        # Extract the text after 'Answer:'
        answer_text = text[index + len('Answer:'):]
        answer = answer_text.strip()

    else:
        answer = "I apologize, I have no knowledge about that"

    return answer


def load_model(model_id):
    model_id = "tiiuae/falcon-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto")

    return pipeline, tokenizer


model_name = "tiiuae/falcon-7b-instruct"
PIPELINE, TOKENIZER = load_model(model_name)

DATA_DIR = "/home/ubuntu/EIP/Damiano/walert/quantitative_eval"


COLLECTION = DATA_DIR + "/data/collection.csv"
TOPICS = DATA_DIR + "/data/topics.csv"
GROUNDTRUTH = DATA_DIR + "/data/groundtruth.csv"
RAG_ANSWERS = DATA_DIR + '/target/runs/rag-bm25.txt'

# Dense Retrieval
# INDEX = "target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
# QUERY_ENCODER = 'facebook/dpr-question_encoder-multiset-base'
# OUTPUT_PATH = 'target/runs/rag-dense-faiss.txt'
# RUN = "dense-faiss"

# searcher = FaissSearcher(
#     INDEX,
#     QUERY_ENCODER
# )

# def get_context_passages(question):
#     # num_hits = 10
#     # hits = searcher.search(question, num_hits)
#     # top_K = 3
#     collection_df = pd.read_csv(COLLECTION)
#     run_rag_df = pd.read_csv(RAG_ANSWERS)
#     topics_df = pd.read_csv(TOPICS)
    
#     rag_passages = []
#     for d in run_rag_df:
#         temp_passage = list(collection_df[collection_df['passage_id'] == d[2]]['passage'])[1]
        
#         context_passages.append(temp_passage)
#     return context_passages

def RAG_context_passages(rag_bm_txt):
    # Read COLLECTION.csv and TOPICS.csv
    collection_df = pd.read_csv(COLLECTION)
    topics_df = pd.read_csv(TOPICS)

    # Read the text file into a DataFrame for easier manipulation
    txt_df = pd.read_csv(rag_bm_txt, sep='\s+', header=None, names=['question_id', 'Qn', 'passage_id', 'N_queries', 'score', 'tag'])

    # Sort the text data by score in descending order
    txt_df = txt_df.sort_values(by=['question_id', 'score'], ascending=[True, False])

    # Keep only the top 3 passages for each question_id
    top_passages_df = txt_df.groupby('question_id').head(3)

    # Merge with the COLLECTION.csv to get passages
    merged_df = top_passages_df.merge(collection_df, how='left', left_on='passage_id', right_on='passage_id')

    # Merge with the TOPICS.csv to get questions
    merged_df = merged_df.merge(topics_df, how='left', left_on='question_id', right_on='question_id')

    # Select only relevant columns for the output
    merged_df = merged_df[['question_id', 'Qn', 'N_queries', 'score', 'tag', 'question', 'passage']]

    # Write the result to the output file
    return merged_df


def generate_answer(question, context, pipeline, tokenizer):
    
    contexts = context[context['question_id'] == question].values.tolist()
    question = contexts[0][-2]
    
    static_prompt = "Generate an answer based on the retrieved documents for the following question."
    prompt_base = static_prompt + "\n Question: " + contexts[0][-2] + "\n Document 1: " + contexts[0][-1] + "\n Document 2: " + \
                  contexts[1][-1] + "\n Document 3: " + contexts[2][-1] + '\n Answer: '

    gen_answer = pipeline(
        prompt_base,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # max_length=800,
        max_new_tokens=100,
        # top_k=2,
        # max_new_tokens=400,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.0,
        repetition_penalty=1.03)

    return gen_answer

RAG_context_passagess = RAG_context_passages(RAG_ANSWERS)

llm_result = generate_answer('W01Q01', RAG_context_passagess, PIPELINE, TOKENIZER)
#print(llm_result)
response_text = get_answer(llm_result[0]['generated_text'])
print(response_text)

# if __name__ == '__main__':
    # logging.info("Starting the voice assistant...")
    # # ************ Query Recording ************

    # samplerate = 44100  # Standard for most microphones
    # channels = 2  # Stereo

    # audio_data = []

    # Start the recording in a new thread
    # stream = sd.InputStream(callback=callback, channels=channels, samplerate=samplerate)
    # with stream:
    #     # Wait for the user to press Enter
    #     input()
    #     EVENT.set()

    # Concatenate the audio data and save it to a temporary WAV file
    # audio = np.concatenate(audio_data)
    # temp_filename = 'user_voice_query.wav'
    # write(temp_filename, samplerate, audio)

    # logging.info("User's voice query successfully recorded and store as user_voice_query.wav")

    # ************ ASR ************
    # model = whisper.load_model("base")
    # result = model.transcribe(temp_filename, fp16=False)
    # question = result["text"]

    # logging.info(f"User's voice query trasncribed: {question}")

    # logging.info(f"Initiating retrieval . . .")
    # ************ Retrieval ************
    # RAG_context_passages = get_context_passages(question)

    # logging.info(f"Retrieval Completed")
    # ************ Response Generation in Text ************
    # logging.info(f"Initiating response generation using Falcon")
    # llm_result = generate_answer(question, RAG_context_passages, PIPELINE, TOKENIZER)

    # response_text = get_answer(llm_result[0]['generated_text'])

    # logging.info(f"Response generation completed (text format): {response_text}")
    # ************ Voice Response ************
    # logging.info(f"Initiating voice response (audio format)")
    # play_voice_response(response_text)

    # logging.info(f"Conversation turn completed.")


