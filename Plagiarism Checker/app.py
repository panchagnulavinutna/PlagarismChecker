import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_documents(directory_path='.'):
    return [os.path.join(directory_path, doc) for doc in os.listdir(directory_path) if doc.endswith('.txt')]

def vectorize(text_documents):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(text_documents).toarray()

def calculate_similarity(matrix, filenames):
    similarities = cosine_similarity(matrix)
    plagiarism_results = set()

    for i, student_a in enumerate(filenames):
        for j, student_b in enumerate(filenames[i+1:]):
            sim_score = similarities[i, j+i+1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)

    return plagiarism_results

def check_plagiarism(directory_path='.'):
    documents = load_documents(directory_path)
    text_data = [open(doc, encoding='utf-8').read() for doc in documents]
    tfidf_matrix = vectorize(text_data)

    return calculate_similarity(tfidf_matrix, documents)

if __name__ == "__main__":
    for data in check_plagiarism():
        print(data)
