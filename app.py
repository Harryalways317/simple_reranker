from copy import deepcopy
from flask import Flask, make_response, jsonify, request
from flask_restful import Resource, Api, reqparse, abort
from sentence_transformers import CrossEncoder

# import uwsgi
# from uwsgidecorators import postfork
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline
import time

app = Flask(__name__)
api = Api(app)
query_model = None
classifier = None
query_tokenizer = None
classifier = None
classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')

# @postfork
# def init_crossen():
# global model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
query_model = "naver/efficient-splade-VI-BT-large-query"
query_tokenizer = AutoTokenizer.from_pretrained(query_model)
query_model = AutoModelForMaskedLM.from_pretrained(query_model)
# og_model = CrossEncoder('cross-encoder/stsb-roberta-large', device='cuda')
# classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')
# print("type:",type(classifier))
import torch


def crossencode(*args, **kwargs):
    # torch.set_num_threads(4)
    start = time.time()

    question = kwargs['question']
    prospective_matches = kwargs['matches']

    encoder_input = [(question, prospective_match) for prospective_match, _ in prospective_matches]
    encoder_output = og_model.predict(encoder_input)
    output_tuple = [(prospective_matches[i][1], float(encoder_output[i])) for i in range(len(prospective_matches))]
    output_tuple = sorted(output_tuple, key=lambda x: x[1], reverse=True)

    end = time.time()
    print("w/o parallel", end - start)
    return output_tuple


def compute_vector(text, model, tokenizer):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = tokenizer(text, return_tensors="pt")
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    return vec, tokens


def extract_and_map_sparse_vector(vector, tokenizer):
    """
    Extracts non-zero elements from a given vector and maps these elements to their human-readable tokens using a tokenizer. The function creates and returns a sorted dictionary where keys are the tokens corresponding to non-zero elements in the vector, and values are the weights of these elements, sorted in descending order of weights.

    This function is useful in NLP tasks where you need to understand the significance of different tokens based on a model's output vector. It first identifies non-zero values in the vector, maps them to tokens, and sorts them by weight for better interpretability.

    Args:
    vector (torch.Tensor): A PyTorch tensor from which to extract non-zero elements.
    tokenizer: The tokenizer used for tokenization in the model, providing the mapping from tokens to indices.

    Returns:
    dict: A sorted dictionary mapping human-readable tokens to their corresponding non-zero weights.
    """

    # Extract indices and values of non-zero elements in the vector
    cols = vector.nonzero().squeeze().cpu().tolist()
    weights = vector[cols].cpu().tolist()

    # Map indices to tokens and create a dictionary
    idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    token_weight_dict = {
        idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
    }

    # Sort the dictionary by weights in descending order
    sorted_token_weight_dict = {
        k: v
        for k, v in sorted(
            token_weight_dict.items(), key=lambda item: item[1], reverse=True
        )
    }

    return sorted_token_weight_dict


class MSMacroAPI(Resource):
    def post(self):
        # Initialize the parser
        parser = reqparse.RequestParser()
        parser.add_argument('question', type=str, required=True, help='Field "question" cannot be empty.')
        parser.add_argument('docs', type=str, action='append', required=True, help='Field "docs" must be a list of document content.')
        args = parser.parse_args()

        # Extract arguments
        question = args['question']
        docs = args['docs']  # This is now a list of document content strings

        # Prepare data for the model
        doc_summaries_ls = [[question, doc] for doc in docs]
        scores = model.predict(doc_summaries_ls)

        # Pair scores with document content
        zip_rel_docs = [[scores[cnt], doc] for cnt, doc in enumerate(docs)]
        zip_rel_docs.sort(key=lambda x: -1 * x[0])

        # Return the top 5 documents by their content
        top_docs = [i[1] for i in zip_rel_docs[:5]]

        # Return the top 5 documents
        return make_response(jsonify({"docs": top_docs}), 200)

class MSURLRerank(Resource):
    def post(self):
        # Initialize the parser for metadata reranking functionality
        parser = reqparse.RequestParser()
        parser.add_argument('question', type=str, required=True, help='Field "question" cannot be empty.')
        parser.add_argument('doc_meta', type=dict, action='append', required=True,
                            help='Field "doc_meta" must be a list of dictionaries with keys "url" and "description".')
        args = parser.parse_args()

        # Extract arguments
        question = args['question']
        doc_meta = args['doc_meta']  # This is now a list of dictionaries

        # Prepare data for the model
        doc_summaries_ls = [[question, meta['description']] for meta in doc_meta]
        scores = model.predict(doc_summaries_ls)
        print("Scores:", scores)

        # Debugging: Print each cnt and meta for clarity
        for cnt, meta in enumerate(doc_meta):
            print("Index:", cnt)
            print("Meta:", meta)

        # Pair scores with URLs using the metadata description
        # Correctly use enumerate to ensure `cnt` is used as an index for scores
        zip_rel_docs = [[scores[cnt], meta['url']] for cnt, meta in enumerate(doc_meta)]
        zip_rel_docs.sort(key=lambda x: -1 * x[0])

        # Return documents in sorted order based on scores
        sorted_docs = [{"url": doc[1]} for doc in zip_rel_docs]  # Create a list of dicts with 'url'

        # Return the sorted documents
        return make_response(jsonify({"docs": sorted_docs}), 200)



api.add_resource(MSMacroAPI, '/msmacro')
api.add_resource(MSURLRerank, '/msurlrerank')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)