# Adobe India Hackathon 2025 - Round 1: Connecting the Dots Through Docs

## Overview

This repository contains the solution for Round 1 of the Adobe India Hackathon 2025, titled "Connecting the Dots Through Docs." The challenge focuses on building intelligent document processing systems to extract, structure, and prioritize information from PDF documents based on various user needs and contexts.

The solution is designed to be modular, efficient, and compliant with strict performance and resource constraints, including CPU-only execution and offline capability.

## Challenges Completed

This solution successfully addresses the requirements of two distinct challenges, with **Challenge 1B's solution building upon the core functionalities developed for Challenge 1A.**

### 1. Challenge 1A: Understand Your Document (Outline Extraction)

**Mission:** To extract a structured outline (Title, H1, H2, H3 with page numbers) from a single PDF document, making its hierarchy machine-understandable.

**Solution Highlights (implemented in `Challenge_1a` and integrated into `Challenge_1b`):**
* **Accurate Title Extraction:** Identifies and consolidates multi-line document titles.
* **Hierarchical Heading Detection:** Employs advanced on-device heuristics (font size, boldness, vertical spacing, content patterns) to classify H1, H2, and H3 headings.
* **Noise Filtering:** Effectively removes page numbers, boilerplate text, and other non-heading elements.
* **Structured JSON Output:** Provides a clean, hierarchical outline in the specified JSON format.

### 2. Challenge 1B: Persona-Driven Document Intelligence (Multi-Collection Analysis)

**Mission:** To act as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of PDFs based on a specific persona and their job-to-be-done.

**Solution Highlights:**
* **Multi-Collection Processing:** Processes multiple distinct document collections, each with its own set of PDFs and input configurations.
* **Persona-Driven Semantic Analysis:** Utilizes a pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`) downloaded during Docker build for offline runtime, to semantically score document sections against dynamic persona roles and job tasks.
* **Intelligent Sectioning & Ranking:** Segments documents into logical sections, infers heading hierarchy, and ranks sections by their semantic relevance to the user's query.
* **Descriptive Section Titles:** Generates informative section titles, even for content blocks without explicit headings.
* **Refined Subsection Analysis:** Provides concise, highly relevant textual snippets for key subsections, leveraging semantic similarity.
* **Structured JSON Output:** Generates detailed output for each collection, including metadata, extracted sections with importance ranks, and refined subsection analysis.

## Core Technologies & Approach

* **PDF Parsing:** `PyMuPDF` (fitz) for high-performance extraction of text, layout, and style information.
* **Natural Language Processing (NLP):** `sentence-transformers` for generating semantic embeddings, enabling advanced relevance scoring.
* **Heuristics & Rule-based Logic:** Custom Python logic for robust heading detection, document segmentation, and content filtering, optimized for diverse PDF layouts.
* **Parallel Processing:** `concurrent.futures.ProcessPoolExecutor` for efficient multi-core utilization during PDF and collection processing.
* **Containerization:** Docker for packaging the solution, ensuring a consistent and isolated execution environment.

## Project Structure (Overall Repository Layout)

project_root/
├── README.md                           # This file (overall project README)
├── Challenge_1a/                       # Directory for Challenge 1A specific files (if kept separate)
│   ├── Dockerfile                      # Docker configuration for Challenge 1A
│   ├── requirements.txt                # Dependencies for Challenge 1A
│   ├── process_pdfs.py                 # Solution script for Challenge 1A
│   ├── input/                          # Sample input PDFs for Challenge 1A
│   └── output/                         # Output JSONs for Challenge 1A
└── Challenge_1b/                       # Directory for Challenge 1B solution (integrates 1A core)
├── Collection 1/
│   ├── PDFs/                       # Input PDF documents for this collection
│   ├── challenge1b_input.json      # Input configuration (persona, job-to-be-done, document list)
│   └── challenge1b_output.json     # Generated output: Persona-driven analysis results
├── Collection 2/
│   ├── PDFs/
│   └── challenge1b_input.json
├── Collection 3/
│   ├── PDFs/
│   └── challenge1b_input.json
├── Dockerfile                      # Docker container configuration for Challenge 1B solution
├── requirements.txt                # Python dependencies for Challenge 1B
├── pdf_extractor.py                # Handles low-level PDF parsing and element extraction (reused from 1A principles)
├── challenge1b_processor.py        # Orchestrates collection processing, semantic analysis, and output generation
└── approach_explanation.md         # Detailed explanation of methodology and design choices

## How to Build and Run the Solution

**Prerequisites:**

* **Docker Desktop:** Installed and running (ensure your local drive where `project_root` resides is enabled for file sharing in Docker Desktop settings).
* **Project Structure:** Your local `project_root` directory must be set up as described above, with both `Challenge_1a` and `Challenge_1b` subdirectories, their respective inputs, and code files correctly placed.

**Building and Running Challenge 1A (if you want to run it separately):**
1.  **Navigate to the Challenge 1A Directory:**
    ```powershell
    cd C:\Users\ashik\project_root\Challenge_1a
    ```
    *(Replace with your actual path)*
2.  **Build the Docker Image:**
    ```powershell
    docker build --platform linux/amd64 -t my1a_solution:latest .
    ```
3.  **Run the Docker Container:**
    ```powershell
    docker run --rm -v "$(Get-Location)/input:/app/input" -v "$(Get-Location)/output:/app/output" --network none my1a_solution:latest
    ```

**Building and Running Challenge 1B (your main integrated solution):**
1.  **Navigate to the Challenge 1B Directory:**
    ```powershell
    cd C:\Users\ashik\project_root\Challenge_1b
    ```
    *(Replace with your actual path)*
2.  **Build the Docker Image:**
    This command builds the Docker image. It includes pre-downloading the `all-MiniLM-L6-v2` Sentence Transformer model during the build process to ensure the solution operates offline during runtime.
    ```powershell
    docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
    ```
    *(Replace `mysolutionname:somerandomidentifier` with your chosen image name and tag. Example: `myhackathonapp:round1`)*
3.  **Run the Docker Container:**
    This command executes the main solution script (`challenge1b_processor.py`) within the Docker container. It mounts your local `Challenge_1b` directory to `/app` inside the container, providing access to all input collections and allowing outputs to be saved back to your local machine.
    ```powershell
    docker run --rm -v "$(Get-Location):/app" mysolutionname:somerandomidentifier
    ```
    *(Ensure you are in the `Challenge_1b` directory when running, and use your chosen image name and tag)*

## Output

Upon successful execution of the Challenge 1B solution, `challenge1b_output.json` files will be generated within each `Collection X/` subdirectory. These JSON files contain the persona-driven analysis results, including metadata, extracted relevant sections with importance ranks, and refined subsection analysis.

**Example `challenge1b_output.json` Structure:**

```json
{
  "metadata": {
    "input_documents": [
      "document1.pdf",
      "document2.pdf"
    ],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "processing_timestamp": "YYYY-MM-DDTHH:MM:SS.microseconds"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Descriptive Section Title from Document",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Concise and semantically relevant snippet of content.",
      "page_number": 1
    }
  ]
}

Models and Libraries Used
PyMuPDF (fitz): Version 1.23.21 - For efficient PDF parsing and extraction of text, layout, and basic structural elements.

sentence-transformers: Version 2.7.0 - For generating semantic embeddings. The all-MiniLM-L6-v2 model is used.

torch: Version 2.3.1 - A deep learning framework, dependency for sentence-transformers.

numpy: For numerical operations, dependency for sentence-transformers.