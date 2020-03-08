# Document Categorization
 Automatic partitioning the collection of my e-books into categories and labeling each category according to its content

The goal of this project was to automatically organize my large collection of e-books in the PDF format into groups or clusters so that I could easily find sources of useful information when needed. However, the code in this repo could be handful for the general purpose where there are many PDF files that one wants to split into groups where each group contains books or documents on similar subject/topic. In machine learning terms, such a task is named document categorization.

Description will be added soon!
## Data
As an example, I used 21 e-books from my personal collection. Here is their list:

These books could roughly be divided into 3-4 clusters.

## Results

## Potential future improvements
I observed that tokens from a programming code sometimes polluted clusters. This happened because many of my books contain a lot of code snippets and text pre-processing, despite being rigorous, was unable to clean up these artifacts. One potential solution of this problem could be paragraph extraction, e.g., based on some heuristics such as blank lines between paragraphs, followed by paragraph classification into code and plain text. Naturally, the latter would require a one-class or binary clssifier trained on examples of code in several popular programming languages and, if a binary classifier is used, plain text. The goal is to filter out paragraphs (almost) entirely consisting of code, while letting paragraphs with minor fraction of code untouched as few instances of code in the whole large book would unlikely result in the high TF-IDF score. 
