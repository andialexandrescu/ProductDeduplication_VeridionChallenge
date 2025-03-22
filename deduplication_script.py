import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time

start_time = time.time()

# `clean_string_data.ipynb`
df = pd.read_parquet('data\\veridion_product_deduplication_challenge.snappy.parquet')
cleaned_df = df.copy()

string_columns = cleaned_df.select_dtypes(include=['object']).columns
for col in string_columns:
    cleaned_df[col] = cleaned_df[col].fillna('').str.lower()
    cleaned_df[col] = cleaned_df[col].replace(r'^\s*$', np.nan, regex=True)

subset_columns = ['product_name', 'product_title', 'product_summary']
for col in subset_columns:
    cleaned_df[col] = cleaned_df[col].str.replace("[^a-zA-Z0-9 ]", "", regex=True)

cleaned_df['page_url'] = cleaned_df['page_url'].apply(lambda x: [x] if not isinstance(x, list) else x)
cleaned_df['root_domain'] = cleaned_df['root_domain'].apply(lambda x: [x] if not isinstance(x, list) else x)
cleaned_df['product_name'] = cleaned_df['product_name'].fillna(cleaned_df['product_title'])
cleaned_df['product_summary'] = cleaned_df['product_summary'].fillna('')
cleaned_df['description'] = cleaned_df['description'].fillna('')

# `filter_data_similarity.ipynb`
vectorizer = TfidfVectorizer(stop_words='english')

# TF-IDF on product_title
def get_top_words(tfidf_matrix, feature_names, threshold=0.5):
    top_words = []
    for i in range(tfidf_matrix.shape[0]):
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        wrds_above_threshold = [feature_names[j] for j, score in enumerate(tfidf_scores) if score > threshold]
        top_words.append(wrds_above_threshold)
    return top_words

tfidf_matrix_title = vectorizer.fit_transform(cleaned_df['product_title'])
feature_names = vectorizer.get_feature_names_out()
cleaned_df['key_identifiers'] = get_top_words(tfidf_matrix_title, feature_names)

# Cosine similarity on product_name
tfidf_matrix_name = vectorizer.fit_transform(cleaned_df['product_name'])
cosine_sim_matrix_name = cosine_similarity(tfidf_matrix_name)

potential_duplicates = []
for i in range(len(cosine_sim_matrix_name)):
    for j in range(i+1, len(cosine_sim_matrix_name)):
        if cosine_sim_matrix_name[i, j] >= 0.85:
            potential_duplicates.append((i, j, cosine_sim_matrix_name[i, j]))

# combine TF_IDF + Cosine similarity
def calculate_keyword_overlap(keywords1, keywords2):
    set1 = set(keywords1)
    set2 = set(keywords2)
    overlap = len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0
    return overlap

validated_duplicates = []
for i, j, similarity in potential_duplicates:
    keywords1 = cleaned_df.iloc[i]['key_identifiers']
    keywords2 = cleaned_df.iloc[j]['key_identifiers']
    
    overlap = calculate_keyword_overlap(keywords1, keywords2)
    if overlap >= 0.5:
        overall_score = 0.2 * similarity + 0.8 * overlap
        if overall_score >= 0.61:
            validated_duplicates.append((i, j, similarity, overlap, overall_score))

# print ("Number of validated duplicates:", len(validated_duplicates))
# print("Validated Duplicates:")
# for i, j, similarity, overlap, overall_score in validated_duplicates:
#     print(f"Product {i} {cleaned_df.iloc[i]['product_title']} ({cleaned_df.iloc[i]['product_name']}) and \nProduct {j} {cleaned_df.iloc[j]['product_title']} ({cleaned_df.iloc[j]['product_name']}):")
#     print(f"\tCosine Similarity: {similarity:.4f}")
#     print(f"\tKeyword Overlap: {overlap:.4f}")
#     print(f"\tOverall Score: {overall_score:.4f}")
#     print()

# `group_duplicates_consolidate_groups.ipynb`
graph = defaultdict(list)
for i, j, _, _, _ in validated_duplicates:
    graph[i].append(j)
    graph[j].append(i)

def find_connected_components(graph):
    visited = set()
    components = []
    for node in graph:
        if node not in visited:
            stack = [node]
            component = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(graph[current])
            components.append(component)
    return components

duplicate_groups = find_connected_components(graph)

all_duplicate_indices = set()
for group in duplicate_groups:
    all_duplicate_indices.update(group)

non_duplicate_indices = set(cleaned_df.index) - all_duplicate_indices
non_duplicates_df = cleaned_df.loc[list(non_duplicate_indices)]

enriched_entries = []
for group in duplicate_groups:
    if len(group) > 1:
        group_rows = cleaned_df.loc[group]
        
        enriched_entry = {
            'unspsc': group_rows['unspsc'].mode()[0],
            'product_title': group_rows['product_title'].mode()[0],
            'product_name': group_rows['product_name'].mode()[0],
            'product_summary': ' '.join(group_rows['product_summary'].dropna().unique()),
            'root_domain': group_rows['root_domain'].explode().unique().tolist(),
            'page_url': group_rows['root_domain'].explode().unique().tolist(),
            'key_identifiers': list(set([keyword for sublist in group_rows['key_identifiers'] for keyword in sublist]))
        }
        enriched_entries.append(enriched_entry)
representatives_df = pd.DataFrame(enriched_entries)

deduplicated_df = pd.concat([non_duplicates_df, representatives_df], ignore_index=True)
deduplicated_df.to_parquet('data\\deduplicated_products.parquet', engine='pyarrow')
print(len(cleaned_df)-len(deduplicated_df))

duration = time.time() - start_time
minutes, seconds = divmod(duration, 60)
print(f"{int(minutes)} minutes and {int(seconds)} seconds")