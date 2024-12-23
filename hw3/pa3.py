from collections import defaultdict
import os
from nltk.stem import PorterStemmer
import re  # Import the 're' module for regular expressions
import math
#import pandas as pd
import csv
# File paths
document_folder = "./data"  
stopword_file = './stopwords.txt'  
training_file = './training.txt'  
stemmer = PorterStemmer()

# Step 1: Read stopwords into a set
with open(stopword_file, 'r') as f:
    stop_words = set(line.strip().lower() for line in f if line.strip())  # Lowercase for uniformity

# Step 2: Tokenization and stemming function
def tokenization(text):
    # 使用正則表達式一次性移除標點和數字
    doc = re.sub(r"[^\w\s]|[\d]", " ", text)  # 移除標點符號和數字
    
    # 將文本轉為小寫
    doc = doc.lower()
    
    # Tokenization: 使用 split() 切分並清理多餘空白
    words = [word.strip() for word in doc.split() if word.strip()]
    
    # Porter's stemming
    stemmer = PorterStemmer()
    stemming = [stemmer.stem(word) for word in words]
    
    # Stop words removal
    token_list = [word for word in stemming if word not in stop_words]
    
    return token_list
    
# Step 3: Preprocess all documents
documents = []  # List to hold processed documents

for i in range(1, 1096):  # Document IDs are 1 to 1095
    file_path = os.path.join(document_folder, f"{i}.txt")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        processed_content = tokenization(content)
        documents.append(processed_content)  # Add to list

# Step 4: Load training.txt and map class_id to document IDs
def load_training_data(training_file):
    """
    Loads the mapping of class IDs to document IDs from training.txt.
    """
    class_docs = defaultdict(list)
    with open(training_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            class_id = int(tokens[0])  # First token is the class ID
            doc_ids = list(map(int, tokens[1:]))  # Remaining tokens are document IDs
            class_docs[class_id].extend(doc_ids)
    return class_docs

class_to_docs = load_training_data(training_file)

def separate_train_test(class_to_docs, documents):
    training_doc_ids = set(doc_id for doc_ids in class_to_docs.values() for doc_id in doc_ids)
    all_doc_ids = set(range(1, len(documents) + 1))  # Assuming document IDs are 1 to N
    test_doc_ids = all_doc_ids - training_doc_ids

    # Group preprocessed training documents by class
    class_documents = defaultdict(list)
    for class_id, doc_ids in class_to_docs.items():
        for doc_id in doc_ids:
            class_documents[class_id].append((doc_id, documents[doc_id - 1]))  # Store doc_id and content

    # Collect preprocessed test documents
    test_documents = [(doc_id, documents[doc_id - 1]) for doc_id in sorted(test_doc_ids)]

    return class_documents, test_documents

class_documents, test_documents = separate_train_test(class_to_docs, documents)



def extract_vocabulary(class_documents):
    vocab = set()
    for doc_ids in class_documents.values():
        for doc_id, doc_content in doc_ids:
            vocab.update(doc_content)
    return vocab


def count_docs(class_documents):
    return sum(len(doc_ids) for doc_ids in class_documents.values())


def calculate_prior(class_documents, total_docs):
    prior = {}
    for class_id, doc_ids in class_documents.items():
        prior[class_id] = len(doc_ids) / total_docs
    return prior

def generate_term_counts(class_documents, svocab):
    term_counts = defaultdict(lambda: defaultdict(int))  # class -> term -> count
    for class_id, doc_ids in class_documents.items():
        for doc_id, doc_content in doc_ids:
            for term in doc_content:
                if term in vocab:
                    term_counts[class_id][term] += 1
    return term_counts

def calculate_conditional_probabilities(term_counts, class_documents, vocab):
    cond_prob = defaultdict(lambda: defaultdict(float))  # cond_prob[class][term] = P(t|c)
    for class_id, term_count in term_counts.items():
        total_terms_in_class = sum(term_count.values())  # Total count of terms in this class
        for term in vocab:
            term_count_in_class = term_count.get(term, 0)
            # Applying Laplace smoothing
            cond_prob[class_id][term] = (term_count_in_class + 1) / (total_terms_in_class + len(vocab))  
    return cond_prob

def count_term_documents(class_documents, vocab):
    term_doc_counts = defaultdict(lambda: defaultdict(int))  # term_doc_counts[class][term] = document count
    for class_id, doc_ids in class_documents.items():
        for doc_id, doc_content in doc_ids:
            for term in set(doc_content):  # 將 doc_content 轉為集合，避免重複計算相同的詞語
                if term in vocab:
                    term_doc_counts[class_id][term] += 1  # Count this term for this class in this document
    return term_doc_counts





def chi2_feature_selection(class_documents, vocab,  term_doc_counts, total_docs, num_features):
    chi2_scores = defaultdict(float)
    N = total_docs
    for term in vocab:

        for class_id, doc_ids in class_documents.items():
            # True Positive (A): term appears in class
            A =  term_doc_counts[class_id].get(term, 0)
            # False Positive (B): term appears in other classes
            B = sum( term_doc_counts[other_class].get(term, 0) 
                    for other_class in class_documents.keys() if other_class != class_id)
            # False Negative (C): term doesn't appear in class
            C = len(doc_ids) - A
            # True Negative (D): term doesn't appear in other classes
            D = N - (A + B + C)

            
            # Calculate expected frequencies
            E_11 = (A + B) * (A + C) / N
            E_10 = (A + B) * (B + D) / N
            E_01 = (C + D) * (A + C) / N
            E_00 = (C + D) * (B + D) / N
            
            # Calculate chi-squared for each cell
            chi2 = 0
            for obs, exp in [(A, E_11), (B, E_10), (C, E_01), (D, E_00)]:
                if exp > 0:
                    chi2 += (obs - exp) ** 2 / exp
            chi2_scores[term] += chi2
    
    # Sort terms by chi-squared score
    sorted_terms = sorted(chi2_scores.items(), key=lambda x: x[1], reverse=True)

    top_terms = set(term for term, value in sorted_terms[:num_features])
    return top_terms

def apply_multinomial_nb(class_documents, vocab, prior, condprob, test_documents):
    def extract_tokens_from_doc(vocab, doc_content):
        return [term for term in doc_content if term in vocab]
    
    predictions = []
    for doc_id, doc_content in test_documents:
        W = extract_tokens_from_doc(vocab, doc_content)
        scores = defaultdict(float)  
       
        for class_id in class_documents.keys():
            score = math.log(prior[class_id])
            for term in W:
                score += math.log(condprob[class_id][term])
            
            scores[class_id] = score

        predicted_class = max(scores, key=scores.get)
        predictions.append((doc_id, predicted_class))
    
    return predictions


vocab = extract_vocabulary(class_documents)
print("Total number of vocabs:", len(vocab))

# Step 2: Count Documents
total_docs = count_docs(class_documents)
print("Total number of documents:", total_docs)

# Step 3: Calculate Prior Probabilities
prior = calculate_prior(class_documents, total_docs)

count_document= count_term_documents(class_documents, vocab)


selected_vocab = chi2_feature_selection(class_documents, vocab, count_document, total_docs, num_features=300)
print("Reduced vocabulary size:", len(selected_vocab))


term_counts = generate_term_counts(class_documents, selected_vocab)

cond_prob = calculate_conditional_probabilities(term_counts, class_documents, selected_vocab)

predictions = apply_multinomial_nb(class_documents, selected_vocab, prior, cond_prob, test_documents)

with open('predictions.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Id', 'Value'])  # Write header
    for doc_id, predicted_class in predictions:
        writer.writerow([doc_id, predicted_class])  

print("Predictions saved to predictions.csv")
