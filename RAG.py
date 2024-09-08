from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from huggingface_hub import hf_hub_download

embedding_path = "abokbot/wikipedia-embedding"

# Load embeddings, encoders, and dataset
def load_embedding():
    print("Loading embedding...")
    path = hf_hub_download(repo_id="abokbot/wikipedia-embedding", filename="wikipedia_en_embedding.pt")
    wikipedia_embedding = torch.load(path, map_location=torch.device('cpu'))
    print("Embedding loaded!")
    return wikipedia_embedding

wikipedia_embedding = load_embedding()

def load_encoders():
    print("Loading encoders...")
    bi_encoder = SentenceTransformer('msmarco-MiniLM-L-6-v3')
    bi_encoder.max_seq_length = 256  # Truncate long passages to 256 tokens
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    print("Encoders loaded!")
    return bi_encoder, cross_encoder

bi_encoder, cross_encoder = load_encoders()

def load_wikipedia_dataset():
    print("Loading wikipedia dataset...")
    dataset = load_dataset("abokbot/wikipedia-first-paragraph")["train"]
    print("Dataset loaded!")
    return dataset

dataset = load_wikipedia_dataset()
def search(query):
    print("Input question:", query)

    ##### Semantic Search #####
    print("Semantic Search")
    # Encode the query using the bi-encoder and find the top passage
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, wikipedia_embedding, top_k=1)[0]

    if not hits:
        return {"yes_no": "No", "cross_score": 0, "title": "", "abstract": "", "link": ""}

    # Use the top hit for re-ranking
    top_hit = hits[0]
    cross_inp = [[query, dataset[top_hit['corpus_id']]["text"]]]
    cross_scores = cross_encoder.predict(cross_inp)

    result = {
        "yes_no": "Yes" if cross_scores[0] > 7.0 else "No",
        "cross_score": round(cross_scores[0], 3),
        "title": dataset[top_hit['corpus_id']]["title"],
        "abstract": dataset[top_hit['corpus_id']]["text"].split('.')[0] + '.',
        "link": dataset[top_hit['corpus_id']]["url"]
    }

    return result
def verification_chain(data: dict):
    split_questions = data['verification_questions'].strip().split("\n")
    answers = [search(q) for q in split_questions]
    final_response = "\n".join([f"Question: {question.strip()} Answer: {answer}" for question, answer in zip(split_questions, answers)])
    return final_response
