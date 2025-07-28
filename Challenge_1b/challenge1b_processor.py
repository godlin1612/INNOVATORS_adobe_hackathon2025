import json
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import datetime
from sentence_transformers import SentenceTransformer, util

# Import the raw PDF extraction logic from pdf_extractor.py
from pdf_extractor import extract_raw_data_from_pdf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Base directory where collections are located (assuming /app is the mount point for Challenge_1b)
BASE_CHALLENGE_DIR = Path("/app") 

# Global model instance (loaded once per process to handle multiprocessing safely)
_sbert_model = None

def get_sbert_model():
    """Loads the SentenceTransformer model (singleton pattern per process)."""
    global _sbert_model
    if _sbert_model is None:
        logging.info(f"Loading SentenceTransformer model...")
        try:
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2") # Model must be pre-downloaded by Dockerfile
            logging.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            raise RuntimeError("Could not load SentenceTransformer model.") # Fatal if model can't load
    return _sbert_model

def calculate_semantic_relevance_score(text_segment, query_embedding):
    """
    Calculates semantic relevance using SentenceTransformer cosine similarity.
    """
    model = get_sbert_model() # Get model instance for this process
    
    if model is None: 
        return 0.0 # Should not happen if get_sbert_model raises RuntimeError on failure
    
    try:
        # Ensure text segment is not empty for encoding
        if not text_segment.strip():
            return 0.0
        segment_embedding = model.encode(text_segment, convert_to_tensor=True)
        query_embedding_tensor = query_embedding # Already a tensor
        score = util.cos_sim(segment_embedding, query_embedding_tensor).item()
        return max(0.0, score) # Clamp score between 0 and 1, as cosine similarity can be negative
    except Exception as e:
        logging.warning(f"Error calculating semantic similarity for segment: {e}. Returning 0.0.", exc_info=True)
        return 0.0


# Helper for keyword extraction (used for fallback scoring or general text analysis)
def get_keywords(text):
    """Simple keyword extraction: split by spaces, remove common punctuation, convert to lower."""
    text = re.sub(r'[^\w\s]', '', text).lower()
    return set(word for word in text.split() if len(word) > 2) # Exclude very short words


# --- Heading Inference Logic (from 1A, adapted for 1B context) ---
def infer_heading_levels(all_pages_elements, doc_title_text):
    if not all_pages_elements:
        return []

    valid_elements = [e for e in all_pages_elements if e.get("font_size") is not None and e["font_size"] > 0 and e.get("text")]
    if not valid_elements:
        return []

    font_size_counts = defaultdict(int)
    for elem in valid_elements:
        font_size_counts[elem["font_size"]] += len(elem["text"])
    
    sorted_font_sizes_by_char_count = sorted(font_size_counts.items(), key=lambda item: item[1], reverse=True)
    body_font_size = sorted_font_sizes_by_char_count[0][0] if sorted_font_sizes_by_char_count else 0

    unique_font_sizes = sorted(list(set(e["font_size"] for e in valid_elements)), reverse=True)
    font_size_rank_map = {size: i for i, size in enumerate(unique_font_sizes)}

    potential_headings = []
    valid_elements.sort(key=lambda e: (e["page"], e["y0"]))
    cleaned_doc_title = doc_title_text.strip() if doc_title_text else ""

    for i, elem in enumerate(valid_elements):
        score = 0
        
        if cleaned_doc_title and elem["text"].strip() == cleaned_doc_title:
            continue

        if elem["font_size"] > body_font_size * 1.3:
            score += 4
        elif elem["font_size"] > body_font_size * 1.1:
            score += 2
        
        if elem["is_bold"]:
            score += 2

        if i > 0:
            prev_elem = valid_elements[i-1]
            vertical_gap = elem["y0"] - prev_elem["y1"]
            if vertical_gap > (prev_elem["font_size"] * 1.8) and vertical_gap < (elem["font_size"] * 4):
                score += 3
        else:
            if elem["page"] > 1 and elem["y0"] < elem["bbox"][3] * 3 and (elem["font_size"] > body_font_size * 1.2 or elem["is_bold"]):
                 score += 2

        if len(elem["text"]) > 120:
            score -= 4
        elif len(elem["text"]) > 70:
            score -= 2
        
        text_upper = elem["text"].strip().upper()
        # Boost common document sections and persona-relevant terms for heading score
        if text_upper in ["BONAFIDE CERTIFICATE", "ACKNOWLEDGEMENT", "ABSTRACT", "TABLE OF CONTENTS", 
                          "LIST OF FIGURES", "LIST OF TABLES", "APPENDIX", "REFERENCES", 
                          "CONCLUSION", "INTRODUCTION"] or \
           text_upper.startswith("CHAPTER") or text_upper.startswith("SECTION") or text_upper.startswith("APPENDIX"):
           score += 5

        if elem["page"] == 1 and ( # Filter out common boilerplate text from page 1
            "SUBMITTED BY" in text_upper or 
            "IN PARTIAL FULFILLMENT" in text_upper or
            "ENGINEERING COLLEGE" in text_upper or
            "ANNA UNIVERSITY" in text_upper or
            "LIVE IN LAB" in text_upper
        ):
            score -= 10
        
        if "CHAPTER NO. TITLE PAGE NO." in text_upper: # Filter out specific noise
            score -= 100

        if score >= 4: # Threshold for considering it a potential heading
            potential_headings.append({
                "text": elem["text"],
                "font_size": elem["font_size"],
                "is_bold": elem["is_bold"],
                "rank_index": font_size_rank_map.get(elem["font_size"], float('inf')),
                "page": elem["page"],
                "bbox": elem["bbox"],
                "y0": elem["y0"]
            })
    
    if not potential_headings:
        return []

    potential_headings.sort(key=lambda h: (h["page"], h["y0"]))

    grouped_headings = []
    if potential_headings:
        current_group = [potential_headings[0]]
        for i in range(1, len(potential_headings)):
            prev_h = potential_headings[i-1]
            curr_h = potential_headings[i]
            is_close_vertically_for_grouping = (curr_h["y0"] - prev_h["bbox"][3]) < (prev_h["font_size"] * 1.0) # Within ~1 line height
            is_same_font_size = curr_h["font_size"] == prev_h["font_size"]
            is_same_boldness = curr_h["is_bold"] == prev_h["is_bold"]

            if curr_h["page"] == prev_h["page"] and is_close_vertically_for_grouping and is_same_font_size and is_same_boldness:
                current_group.append(curr_h)
            else:
                grouped_headings.append(current_group)
                current_group = [curr_h]
        grouped_headings.append(current_group) # Add the last group

    final_outline_elements = []
    for group in grouped_headings:
        merged_text = " ".join([h["text"] for h in group])
        representative_heading = group[0] 
        
        final_outline_elements.append({
            "text": merged_text,
            "font_size": representative_heading["font_size"],
            "is_bold": representative_heading["is_bold"],
            "rank_index": representative_heading["rank_index"],
            "page": representative_heading["page"],
            "y0": representative_heading["y0"]
        })
    
    unique_ranks = sorted(list(set(h["rank_index"] for h in final_outline_elements)))
    
    level_map = {}
    if len(unique_ranks) >= 1: level_map[unique_ranks[0]] = "H1"
    if len(unique_ranks) >= 2: level_map[unique_ranks[1]] = "H2"
    if len(unique_ranks) >= 3: level_map[unique_ranks[2]] = "H3"
    
    for i in range(3, len(unique_ranks)):
        level_map[unique_ranks[i]] = "H3" 

    final_outline = []
    for h in final_outline_elements:
        level = level_map.get(h["rank_index"], "H4") # Default to H4 for non-H1/H2/H3 headings/content

        text_upper = h["text"].strip().upper()
        if text_upper in ["BONAFIDE CERTIFICATE", "ACKNOWLEDGEMENT", "ABSTRACT", "TABLE OF CONTENTS", 
                          "LIST OF FIGURES", "LIST OF TABLES", "REFERENCES", "CONCLUSION", "INTRODUCTION",
                          # Challenge 1B specific keywords, explicitly force H1 if they appear as headings
                          "FINANCIAL OVERVIEW", "TECHNICAL IMPLEMENTATION MILESTONES",
                          "FUTURE SCOPE", "COMPLIANCE & REGULATIONS", "INTELLECTUAL PROPERTY",
                          "KEY PERFORMANCE INDICATORS (KPI SECTION)", "KEY PERFORMANCE INDICATORS"] or \
           text_upper.startswith("CHAPTER") or text_upper.startswith("SECTION") or text_upper.startswith("APPENDIX"):
            level = "H1"
            
        final_outline.append({
            "level": level,
            "text": h["text"],
            "page": h["page"]
        })
    
    if len(final_outline) > 1:
        # Collect font sizes assigned to H1-H3
        assigned_h1_h2_h3_fonts = sorted(list(set(h["font_size"] for h in final_outline_elements if h.get("level") in ["H1", "H2", "H3"])), reverse=True) 
        
        if assigned_h1_h2_h3_fonts:
            largest_h1_font = assigned_h1_h2_h3_fonts[0]
            
            potential_lower_level_fonts = sorted([
                fs for fs in unique_font_sizes 
                if fs < largest_h1_font and fs >= body_font_size * 1.05 # Must be larger than body text
            ], reverse=True)

            for item in final_outline:
                elem_font_size = next((e["font_size"] for e in final_outline_elements if e["text"] == item["text"] and e["page"] == item["page"]), 0)
                
                if item["level"] != "H1": # Only re-evaluate if not already a confident H1
                    if elem_font_size == largest_h1_font: # If it has the same font size as a confirmed H1, it should be H1
                        item["level"] = "H1"
                    elif len(potential_lower_level_fonts) >= 1 and elem_font_size == potential_lower_level_fonts[0]:
                        item["level"] = "H2"
                    elif len(potential_lower_level_fonts) >= 2 and elem_font_size == potential_lower_level_fonts[1]:
                        item["level"] = "H3"
                    elif elem_font_size > body_font_size * 1.05: # Anything significantly larger than body is at least H3
                        item["level"] = "H3"
                    else: 
                        if item["level"] not in ["H1", "H2", "H3"]: 
                            item["level"] = "H4"
    return final_outline


def segment_document_into_sections(all_elements, outline):
    """
    Segments the document into logical sections based on detected headings and content blocks.
    Each section will have a title (if a heading), its content, and page range.
    """
    segmented_sections = []
    current_section = None
    
    all_elements.sort(key=lambda e: (e["page"], e["y0"])) # Ensure elements are ordered

    # Create a map for quick lookup of outline headings for efficient stopping
    outline_map = {}
    for h in outline:
        key = (h["page"], h["text"].strip().lower())
        outline_map[key] = h # Store the full heading element to get its level

    for i, elem in enumerate(all_elements):
        is_heading_from_outline = False
        heading_level = None
        
        # Check if the current element is a heading from our outline
        outline_key = (elem["page"], elem["text"].strip().lower())
        if outline_key in outline_map:
            is_heading_from_outline = True
            heading_level = outline_map[outline_key]["level"]

        if is_heading_from_outline:
            # If a new heading is found, finalize the previous section (if any)
            if current_section:
                segmented_sections.append(current_section)
            
            # Start a new section with this heading
            current_section = {
                "section_title": elem["text"],
                "heading_level": heading_level, # Use the inferred level
                "document_content": [elem["text"]], # Start content with the heading text itself
                "start_page": elem["page"],
                "end_page": elem["page"]
            }
        else:
            # Add element content to the current section
            if current_section:
                current_section["document_content"].append(elem["text"])
                current_section["end_page"] = elem["page"]
            else:
                # Handle initial content before any heading
                # If there's content before any identified heading, create an "un-titled" section
                if not segmented_sections and not current_section: 
                    current_section = {
                        "section_title": "Document Introduction", # Default title for leading content
                        "heading_level": "H4", # Low level for non-heading content
                        "document_content": [elem["text"]],
                        "start_page": elem["page"],
                        "end_page": elem["page"]
                    }
                elif segmented_sections: # Append to the last section if no new heading was found
                    segmented_sections[-1]["document_content"].append(elem["text"])
                    segmented_sections[-1]["end_page"] = elem["page"]
                # If current_section is None and segmented_sections is not empty, means previous was a table or malformed.
                # In this specific case, elements might be lost if they are not headings and don't fall under a section.
                # For this challenge, we assume content always follows a section or is intro content.

    if current_section: # Add the last section
        segmented_sections.append(current_section)
    
    # Merge document_content into a single string for each section
    for section in segmented_sections:
        section["document_content"] = " ".join(section["document_content"]).strip()
        # Clean up common PDF extraction artifacts and normalize whitespace
        section["document_content"] = re.sub(r'\s+', ' ', section["document_content"]).strip()

        # --- NEW SECTION TITLE INFERENCE LOGIC ---
        # If the current title is generic, try to infer a better one from content
        generic_titles = ["Document Introduction", "Introduction", "Conclusion", ""]
        if section["section_title"] in generic_titles and section["document_content"]:
            # Split into sentences to try to find a concise title
            sentences = re.split(r'(?<=[.!?])\s+', section["document_content"])
            if sentences and len(sentences[0].strip()) > 10 and len(sentences[0].strip()) < 100: # Heuristic for title length
                section["section_title"] = sentences[0].strip()
            elif section["document_content"].strip(): # Fallback to first few words if no good sentence
                words = section["document_content"].split()
                section["section_title"] = " ".join(words[:15]).strip() + "..." if len(words) > 15 else " ".join(words).strip()
            
            # If after all this, it's still generic/empty, re-assign a default descriptive title
            if section["section_title"].strip() in generic_titles or not section["section_title"].strip():
                if section["heading_level"] == "H4":
                    section["section_title"] = f"Content from Page {section['start_page']}"
                else: # For other generic headings like "Introduction" that weren't replaced
                    section["section_title"] = f"{section['heading_level']} from Page {section['start_page']}"
        # --- END NEW SECTION TITLE INFERENCE LOGIC ---

    # Filter out empty sections
    return [s for s in segmented_sections if s["document_content"].strip()]


def process_collection(collection_path: Path):
    """Processes a single collection folder."""
    collection_name = collection_path.name
    logging.info(f"Processing collection: {collection_name}")

    # Define paths within the current collection - MOVED HERE
    input_json_path = collection_path / "challenge1b_input.json"
    output_json_path = collection_path / "challenge1b_output.json"
    pdfs_dir = collection_path / "PDFs"

    if not input_json_path.exists():
        logging.error(f"Input JSON not found for {collection_name} at {input_json_path}. Skipping.")
        return

    if not pdfs_dir.exists():
        logging.error(f"PDFs directory not found for {collection_name} at {pdfs_dir}. Skipping.")
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona_role = input_data["persona"]["role"]
    job_task = input_data["job_to_be_done"]["task"]
    challenge_id = input_data["challenge_info"]["challenge_id"]

    # Combine persona and job for a single query embedding
    combined_query_text = f"Persona: {persona_role}. Task: {job_task}"
    query_embedding = None
    try:
        local_model = get_sbert_model() 
        query_embedding = local_model.encode(combined_query_text, convert_to_tensor=True)
        logging.info(f"Generated query embedding for '{collection_name}'.")
    except Exception as e:
        logging.error(f"Failed to encode query text for '{collection_name}': {e}. Semantic scoring will be disabled for this collection.", exc_info=True)
        query_embedding = None


    all_collection_sections = []
    input_document_names = []

    pdf_files_in_config = []
    # If documents are specified in input_config, use those, else glob all PDFs in the folder
    if input_data.get("documents"):
        for doc_info in input_data["documents"]:
            filename = doc_info["filename"]
            pdf_file_path = pdfs_dir / filename
            if pdf_file_path.exists():
                pdf_files_in_config.append(pdf_file_path)
            else:
                logging.warning(f"Document '{filename}' specified in input JSON not found at '{pdf_file_path}'. Skipping this document.")
    else:
        logging.warning(f"No specific documents listed in input JSON for {collection_name}. Processing all PDFs in {pdfs_dir}.")
        pdf_files_in_config = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files_in_config:
        logging.warning(f"No PDF files found to process for {collection_name}.")
        return

    logging.info(f"Found {len(pdf_files_in_config)} PDF(s) to process for {collection_name}.")

    for pdf_file in pdf_files_in_config: # Loop through identified PDFs
        logging.info(f"Analyzing PDF: {pdf_file.name} in {collection_name}")
        input_document_names.append(pdf_file.name)
        
        # Raw data extraction from pdf_extractor.py
        raw_pdf_output = extract_raw_data_from_pdf(pdf_file)
        if raw_pdf_output is None:
            logging.error(f"Raw PDF extraction failed for {pdf_file.name}. Skipping.")
            continue

        all_elements_from_pdf = raw_pdf_output["all_elements"]
        doc_title = raw_pdf_output["doc_title"]
        
        # Infer headings for the current PDF (using the refined 1A logic)
        outline = infer_heading_levels(all_elements_from_pdf, doc_title)
        
        # Segment the PDF into logical sections based on the outline
        pdf_sections = segment_document_into_sections(all_elements_from_pdf, outline)

        # Calculate relevance for each section
        for section in pdf_sections:
            section_text = section["document_content"]
            
            relevance_score = 0.0
            if query_embedding is not None and len(section_text.strip()) > 50: # Minimum length for meaningful semantic embedding
                relevance_score = calculate_semantic_relevance_score(section_text, query_embedding)
            else:
                # Fallback to simple keyword relevance if semantic model is not used or content too short
                # This ensures some scoring even without the model
                query_keywords = get_keywords(combined_query_text)
                section_keywords = get_keywords(section_text)
                if query_keywords:
                    relevance_score = len(section_keywords.intersection(query_keywords)) / len(query_keywords) # Simple Jaccard
            
            if relevance_score > 0.01: # Threshold for inclusion, tune as needed
                all_collection_sections.append({
                    "document": pdf_file.name,
                    "section_title": section["section_title"] if section["section_title"] else f"Content from Page {section['start_page']}",
                    "page_number": section["start_page"],
                    "relevance_score": relevance_score,
                    "full_content": section["document_content"], # Keep full content for subsection analysis
                    "heading_level": section["heading_level"] 
                })
        
    # Sort and rank all collected sections from all PDFs in this collection
    # Primary sort by relevance, then by heading level (H1 > H2 > H3 > H4), then by page number
    heading_level_priority = {"H1": 1, "H2": 2, "H3": 3, "H4": 4, "": 5, None: 5} 
    all_collection_sections.sort(key=lambda x: (x["relevance_score"], heading_level_priority.get(x["heading_level"], 5), x["page_number"]), reverse=True)
    
    extracted_sections_output = []
    subsection_analysis_output = []
    
    MAX_SECTIONS_TO_EXTRACT = 100 

    for i, section in enumerate(all_collection_sections[:MAX_SECTIONS_TO_EXTRACT]):
        rank = i + 1
        extracted_sections_output.append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": rank,
            "page_number": section["page_number"]
        })

        sentences = re.split(r'(?<=[.!?])\s+', section["full_content"])
        
        refined_sentences = []
        local_model_for_sentences = get_sbert_model() 
        
        if local_model_for_sentences and query_embedding is not None and sentences:
            try:
                sentence_embeddings = local_model_for_sentences.encode(sentences, convert_to_tensor=True)
                sentence_scores = util.cos_sim(sentence_embeddings, query_embedding).flatten()
                
                scored_sentences = sorted(zip(sentence_scores, sentences), key=lambda x: x[0], reverse=True)
                
                char_limit = 500 
                current_char_count = 0
                for score, sentence in scored_sentences:
                    if current_char_count + len(sentence) <= char_limit and score > 0.1: 
                        refined_sentences.append(sentence)
                        current_char_count += len(sentence)
                    else:
                        break
                
                refined_text = " ".join(refined_sentences).strip()
                if not refined_text and section["full_content"].strip(): 
                    refined_text = " ".join(section["full_content"].split()[:50]) 
            except Exception as e:
                logging.warning(f"Error during sentence embedding/scoring for refined text: {e}. Falling back to first words.", exc_info=True)
                refined_text = " ".join(section["full_content"].split()[:50])
        else:
            refined_text = " ".join(section["full_content"].split()[:50]) 
            
        subsection_analysis_output.append({
            "document": section["document"],
            "refined_text": refined_text,
            "page_number": section["page_number"] 
        })
    
    output_data = {
        "metadata": {
            "input_documents": input_document_names, 
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis_output
    }

    try:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved combined output for '{collection_name}' to '{output_json_path.name}'.")
    except Exception as e:
        logging.error(f"Error saving output JSON for {collection_dir.name}: {e}", exc_info=True)


# --- Main Entry Point ---
def main():
    logging.info("Starting Challenge 1B: Persona-Driven Document Intelligence solution...")
    
    BASE_CHALLENGE_DIR.mkdir(parents=True, exist_ok=True)

    # Dynamically find collection folders (e.g., Collection 1, Collection 2, etc.)
    collection_paths = [d for d in BASE_CHALLENGE_DIR.iterdir() if d.is_dir() and re.match(r"Collection \d+", d.name)]
    
    if not collection_paths:
        logging.warning(f"No collection folders (e.g., 'Collection 1') found in {BASE_CHALLENGE_DIR}.")
        return

    logging.info(f"Found {len(collection_paths)} collection(s) to process.")

    # Process each collection in parallel. Each process will load its own SBERT model instance.
    with ProcessPoolExecutor(max_workers=3) as executor: 
        futures = {executor.submit(process_collection, collection_path): collection_path for collection_path in collection_paths}
        
        for future in as_completed(futures):
            collection_path = futures[future]
            try:
                future.result() 
            except Exception as exc:
                logging.error(f"Collection processing for '{collection_path.name}' generated an exception: {exc}", exc_info=True)

    logging.info("Challenge 1B solution finished.")

if __name__ == "__main__":
    main()
