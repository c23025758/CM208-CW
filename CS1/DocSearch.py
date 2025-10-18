import numpy as np
import math
import multiprocessing
from collections import defaultdict

def read_file(filepath):
    try:
        with open(filepath, "r") as file:
            return file.read().splitlines()
    except FileNotFoundError as e:
        print(e)
        return []

def build_document_dict(documents):
    document_dict = {}
    for idx, doc in enumerate(documents, 1):
        words = doc.split()
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        document_dict[idx] = dict(word_count)
    return document_dict

def build_corpus_dict(documents):
    corpus_dict = defaultdict(int)
    for doc in documents:
        words = doc.split()
        for word in words:
            corpus_dict[word] += 1
    return corpus_dict

def build_inverted_index(documents, corpus_dict):
    inverted_index = defaultdict(list)
    for idx, doc in enumerate(documents, 1):
        words = doc.split()
        word_set = set(words)
        for word in word_set:
            inverted_index[word].append(idx)
    return inverted_index

def search_documents(query, inverted_index, documents, corpus_dict):
    print(f"Query: {query}")
    query_words = set(query.split())
    relevant_documents = set.intersection(*[set(inverted_index[word]) for word in query_words if word in inverted_index])

    if not relevant_documents:
        print("Relevant documents: ")
        return

    angles = {}
    query_vector = np.array([1 if word in query_words else 0 for word in corpus_dict.keys()])
    for doc_id in relevant_documents:
        doc_vector = np.array([documents[doc_id].get(word, 0) for word in corpus_dict.keys()])
        angles[doc_id] = calculate_angle(doc_vector, query_vector)

    sorted_documents = sorted(angles.items(), key=lambda x: x[1], reverse=True)
    print("Relevant documents:", end=" ")
    print(*[doc_id for doc_id, angle in sorted_documents], sep=" ")
    for doc_id, angle in sorted_documents:
        print(f"{doc_id} {angle:.2f}")


def calculate_angle(doc_vector, query_vector):
    norm_doc = np.linalg.norm(doc_vector)
    norm_query = np.linalg.norm(query_vector)
    dot_product = np.dot(doc_vector, query_vector)
    cos_theta = dot_product / (norm_doc * norm_query)
    angle = math.degrees(math.acos(cos_theta))
    return angle

if __name__ == "__main__":
    documents = read_file("docs.txt")
    queries = read_file("queries.txt")

    corpus_dict = build_corpus_dict(documents)
    document_dict = build_document_dict(documents)
    inverted_index = build_inverted_index(documents, corpus_dict)

    print(f"Words in dictionary: {len(corpus_dict)}")

    for query in queries:
        search_documents(query, inverted_index, document_dict, corpus_dict)
