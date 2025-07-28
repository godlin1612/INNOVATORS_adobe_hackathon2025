Approach Explanation: Persona-Driven Document Intelligence (Challenge 1B)
1. Introduction
This document details the methodology and technical approach implemented for Challenge 1B of the Adobe India Hackathon 2025, focusing on persona-driven document intelligence. The solution aims to accurately extract and prioritize relevant information from diverse PDF document collections based on dynamic user personas and their specific tasks. This system is designed to operate efficiently within strict resource constraints, including CPU-only execution and offline capability.

2. Overall Architecture and Modularity
The solution is structured into a modular pipeline to ensure clarity, maintainability, and reusability of components. It leverages two main Python scripts:

pdf_extractor.py: Dedicated to low-level PDF parsing and raw element extraction.

challenge1b_processor.py: Orchestrates the collection processing, semantic analysis, relevance ranking, and final output generation.

This modularity allows for clear separation of concerns and facilitates independent testing and refinement of each stage.

3. PDF Content and Structure Extraction (pdf_extractor.py)
a. Raw Element Extraction
Library Used: PyMuPDF (fitz) is employed for its high performance and ability to extract not just raw text, but also rich metadata about each text block and tables.

Process: Each page is processed to extract blocks using page.get_text("dict"). Each block provides text content, bounding box (bbox), font size (font_size), and boldness (is_bold).

Noise Filtering: Basic heuristics are applied to filter out common document noise such as page numbers (based on position and numeric content) and boilerplate text (based on content on the first page).

b. Table Detection and Extraction
Method: page.find_tables() (PyMuPDF's built-in table detection) is used to identify tables.

Content Retrieval: For each detected table, table.extract() is utilized to pull cell contents (headers and rows) as lists of strings, ensuring structured table data.

Overlap Handling: Text blocks overlapping with identified table bounding boxes are explicitly excluded from general text processing to prevent redundant or malformed extractions.

Consistent Element Structure: A critical design choice was ensuring all extracted elements (text paragraphs/headings, and tables) conform to a unified dictionary structure containing keys like text, font_size, is_bold, page, y0, and y1 for consistent downstream processing. For tables, a text representation of their content is generated for the text field.

4. Document Segmentation and Heading Inference (challenge1b_processor.py)
a. Document Segmentation
Goal: To break the stream of raw elements from a PDF into logical sections suitable for relevance scoring.

Methodology: The document is segmented based on the presence of inferred headings. A new section is initiated whenever a heading is detected. Content following a heading is grouped into that section until the next heading appears.

Initial Content Handling: Content appearing before the very first detected heading is grouped into a default "Document Introduction" section.

b. Hierarchical Heading Inference
Objective: To identify and assign hierarchical levels (H1, H2, H3, H4) to actual headings within the document.

Heuristics Applied (Scoring-based System): Each text block is assigned a score based on multiple indicators:

Font Size: Significantly larger font sizes relative to the document's estimated body_font_size contribute positively.

Boldness: Bold text is a strong indicator of a heading.

Vertical Spacing: Larger vertical gaps above a text block often signify a new section start.

Length: Shorter, more concise text blocks are favored as headings (long sentences are penalized).

Keyword Boosting: Specific common section titles (e.g., "ABSTRACT", "CHAPTER", "APPENDIX", "INTRODUCTION") are given a high positive score to ensure their inclusion and correct leveling.

Noise Suppression: Specific boilerplate text (e.g., "SUBMITTED BY" on cover pages, "CHAPTER NO. TITLE PAGE NO.") is heavily penalized to filter it out.

Grouping Multi-Line Headings: Consecutive text blocks on the same page with similar style (font size, boldness) and very close vertical proximity are merged into a single logical heading.

Level Assignment: Heading levels (H1, H2, H3) are dynamically assigned based on the relative rank_index (derived from font size) of the grouped headings. A post-processing step ensures that major sections identified by keywords (e.g., "CHAPTER X", "APPENDIX Y") are consistently assigned as H1s, and other levels are adjusted accordingly to maintain hierarchy. Generic content blocks are typically assigned 'H4'.

Descriptive section_title Generation: For sections that don't begin with a clear, extracted heading, the first informative sentence or a relevant snippet from its content is extracted and assigned as the section_title, improving output readability.

5. Persona-Driven Relevance Scoring and Ranking
a. Query Embedding
Method: The persona's role and the job_to_be_done's task (from challenge1b_input.json) are combined into a single query string.

Model: A pre-trained SentenceTransformer model (all-MiniLM-L6-v2) is used to generate a semantic embedding (vector representation) of this combined query.

b. Section Relevance Scoring
Method: Each segmented section's document_content is also encoded into a semantic embedding.

Score Calculation: The cosine similarity between the section's embedding and the query embedding is calculated using util.cos_sim. This score quantifies the semantic relevance of the section to the persona's needs.

Fallback: For very short sections or if the model loading fails, a simpler keyword overlap (Jaccard similarity) fallback is implemented to ensure some relevance scoring.

c. Ranking and Filtering
Sorting: All sections across all PDFs within a collection are sorted primarily by their relevance_score (descending). A secondary sort prioritizes sections under higher heading_levels (H1 > H2 > H3 > H4), ensuring that more structured and prominent relevant content appears higher in the ranks.

Thresholding: Only sections with a relevance_score above a defined threshold (e.g., 0.01) are included in the final output, filtering out irrelevant content.

Importance Rank Assignment: The sorted sections are assigned a sequential importance_rank.

6. Subsection Analysis and Refined Text Generation
Goal: To provide a concise and highly relevant summary (refined_text) for each extracted section.

Methodology: Within each ranked section's full_content:

The content is split into individual sentences.

Each sentence is then semantically encoded, and its similarity to the original combined persona/job query is calculated.

The top-scoring sentences are selected until a defined character limit (e.g., 500 characters) is met. This acts as an extractive summary.

Fallback: If semantic sentence extraction is not possible (e.g., very short section, model error), the first few words of the section are used as a fallback.

7. Dockerization and Constraints Compliance
Docker Setup: The solution is containerized using a Dockerfile based on python:3.10-slim.

Offline Execution: The all-MiniLM-L6-v2 Sentence Transformer model is pre-downloaded during the Docker image build process (RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(...)"). This ensures that the solution operates completely offline during runtime, meeting the hackathon constraint.

CPU-Only Runtime: All libraries and models used (PyMuPDF, Sentence Transformers) are configured for CPU execution, avoiding any GPU dependencies.

Resource Management: Python's concurrent.futures.ProcessPoolExecutor is used for parallel processing of collections and individual PDFs, effectively utilizing multiple CPU cores and managing memory by processing documents in isolation. The model is loaded once per process to optimize resource use.

Time Compliance: The selection of efficient libraries and optimized algorithms aims to keep processing time within the specified limits (e.g., ≤60 seconds for 3-5 documents).