# ProductDeduplication_VeridionChallenge

## Project summary
This project aims consolidate duplicates into a single, enriched entry per product, maximizing available information while ensuring uniqueness. The dataset contains product details extracted from various web pages using LLMs, resulting in duplicate entries where the same product appears across different sources. Each row represents partial attributes of a product. The resource for this project is a .parquet file [ veridion_product_deduplication_challenge.snappy.parquet file](https://github.com/andialexandrescu/ProductDeduplication_VeridionChallenge/blob/main/data/veridion_product_deduplication_challenge.snappy.parquet).

## Important details
Each main part of this project is documented in its corresponding Jupyter Notebook. For detailed implementation, please refer to the .ipynb files, since this README serves the purpose of being an introduction.

## Requirements
**Virtual environment**
   - tested with python 3.12.3
   - create an environment called `venv` using `python -m venv ./venv` (or decide upon another name)
   - import the necessary modules listed in the [requirements.txt](https://github.com/andialexandrescu/ProductDeduplication_VeridionChallenge/blob/main/requirements.txt) running the command `pip install -r .\requirements.txt` or `pip install pandas numpy thefuzz scikit-learn pyarrow fastparquet` (only after entering the environment using `.\venv\Scripts\activate` or any alternative command)

## Compilation

Only run the `deduplication_script.py` file and check the Jupyter Notebooks for the implementation logic.

The `deduplication_script.py` file incorporates everything worth keeping from the notebooks and the results of deduplicated products are saved in the [deduplicated_products.parquet](https://github.com/andialexandrescu/ProductDeduplication_VeridionChallenge/blob/main/data/deduplicated_products.parquet).

## Implementation Overview 

I structured the project into **three tasks**, each being implemented in a separate Jupyter Notebook:

1. **Data Preparation**: Cleaned data to ensure consistency
    - Consisted of analysing the data, how it's structured, normalization and tokenization
        - **Immediate insights**: Recognized the necessity of lowercase transformations, stop words and non-alphanumeric characters removal

2. **Similarity Analysis (multi-layered)**: Applied NLP techniques to identify duplicate pairings
    - I applied various NLP algorithms on product columns such as `product_name`, `product_title`, `product_summary` and `root_domain`
        - Used **TF-IDF** for `product_title` in order to identify the most significant words for each product title and combined this metric with **Cosine similarity** applied onto `product_name`
            - **Purpose**: **TF-IDF** captures distinctive features of each product title, while **Cosine similarity** provides a more general comparison of product names as whole sentences, together helping identify related products, even if their descriptions vary in complexity or detail, improving the accuracy of grouping products within similar categories (`product_name`)
        - Used **Levenshtein distance** for `root_domain` and implemented **Cosine similarity** for `product_summary`
            - **Purpose**: Websites selling similar product categories often share overlap in their listed summaries for products, increasing the likelihood of duplicate matches, for this reason, any similarity for `product_summary` entries need to be backed up by a category corresponding `root_domain`
        - **Drawbacks**: My use of NLP algorithms defines a lexical approach instead of a semantic one, that could've been implemented using FastText since it is able to recognize OOV words, unlike other predictive word embeddings and collections models (such as GloVe and Word2Vec)

3. **Entity Consolidation**: Clustered these obtained matches into groups, enriched entities, and reconstructed a deduplicated dataset
    - Previous validated matches are processed into groups by finding the contents of connected components in an undirected graph using DFS and then used for determining entities supposed to replace these duplicates
    - The enriched entities are combined with the non duplicate entries into a new data frame, converted back into a parquet file

## Results
   - Total time for running [`deduplication_script.py`](https://github.com/andialexandrescu/ProductDeduplication_VeridionChallenge/blob/main/deduplication_script.py): 2 minutes
   - 2415 of duplicates removed from 21946 entries
   - Combining all experimental approaches from the notebooks would result in significantly increased execution time
   - Worked on-and-off for this project because of a tight schedule and spent no more than 35 hours overall
   - I am definitely considering improving this by using FastText in order to deal with a predictive model for word embeddings