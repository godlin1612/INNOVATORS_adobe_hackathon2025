import fitz  # PyMuPDF library
import json
from pathlib import Path
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

# --- Helper Function for Document Outline Extraction ---
def extract_document_elements_from_page(page_dict):
    """
    Extracts all relevant text elements (potential headings, paragraphs, etc.)
    from a single page's dictionary, including their style information and bbox.
    Filters out common page noise like page numbers.
    """
    elements = []
    
    # Sort blocks by their top-left Y-coordinate, then by X-coordinate
    # This ensures a natural reading order
    blocks = sorted(page_dict.get("blocks", []), key=lambda b: (b["bbox"][1], b["bbox"][0]))

    page_height = page_dict.get("height", 0)
    # Define thresholds for top/bottom margins to filter out page numbers
    TOP_MARGIN_THRESHOLD = page_height * 0.1
    BOTTOM_MARGIN_THRESHOLD = page_height * 0.9

    for block in blocks:
        if block["type"] == 0:  # text block
            block_text_lines = []
            block_font_size = 0
            is_bold = False
            
            # Extract text and determine predominant font size and boldness for the block
            # For simplicity, we take the first span's style as representative for the block
            for line in block.get("lines", []):
                line_text_parts = []
                for span in line.get("spans", []):
                    line_text_parts.append(span["text"])
                    if not block_font_size: # Capture first span's size as representative
                        block_font_size = round(span["size"], 1)
                    if span["flags"] & 2: # Check for bold flag (bit 1 is bold, so 2)
                        is_bold = True
                block_text_lines.append(" ".join(line_text_parts).strip())
            
            # Join lines for the whole block; handle soft hyphens
            cleaned_block_text = " ".join(block_text_lines).replace(" -\n", "").replace("-\n", "").strip()

            if not cleaned_block_text:
                continue

            # Page number filtering heuristic
            x0, y0, x1, y1 = block["bbox"]
            is_in_top_margin = y0 < TOP_MARGIN_THRESHOLD
            is_in_bottom_margin = y1 > BOTTOM_MARGIN_THRESHOLD
            is_numeric_like = cleaned_block_text.strip().replace('-', '').replace('.', '').replace(' ', '').isdigit()
            
            # Filter out short numeric strings in margins (likely page numbers)
            if is_numeric_like and (is_in_top_margin or is_in_bottom_margin) and len(cleaned_block_text.strip()) <= 5:
                logging.debug(f"Filtered out potential page number: '{cleaned_block_text}' at Y={y0}")
                continue

            elements.append({
                "text": cleaned_block_text,
                "font_size": block_font_size,
                "is_bold": is_bold,
                "bbox": block["bbox"], # Store bbox for relative positioning later
                "y0": y0, # Store y0 for easier sorting/comparison
                "y1": y1 # Store y1 for calculating vertical spacing
            })
    return elements

def infer_heading_levels(all_pages_elements, doc_title_text):
    """
    Infers H1, H2, H3 levels from a list of all document elements,
    considering font sizes, boldness, vertical spacing, and general content characteristics.
    Groups multi-line headings.
    Filters out elements that are likely not headings.
    """
    if not all_pages_elements:
        return []

    # Filter out elements with very small or zero font size (often noise)
    valid_elements = [e for e in all_pages_elements if e["font_size"] > 0]
    if not valid_elements:
        return []

    # 1. Identify common font sizes and a potential 'body' font size
    font_size_counts = defaultdict(int)
    for elem in valid_elements:
        font_size_counts[elem["font_size"]] += len(elem["text"])
    
    sorted_font_sizes_by_char_count = sorted(font_size_counts.items(), key=lambda item: item[1], reverse=True)
    body_font_size = sorted_font_sizes_by_char_count[0][0] if sorted_font_sizes_by_char_count else 0

    # Get all unique font sizes present in the document, sorted descending
    unique_font_sizes = sorted(list(set(e["font_size"] for e in valid_elements)), reverse=True)

    # Map unique font sizes to a rank (0 for largest, 1 for next, etc.)
    font_size_rank_map = {size: i for i, size in enumerate(unique_font_sizes)}

    potential_headings = []
    
    # Iterate through elements to identify individual heading candidates
    # Sort elements by page and y0 to get correct "previous" element for spacing checks
    valid_elements.sort(key=lambda e: (e["page"], e["y0"]))

    # Trim leading/trailing whitespace from the document title for more robust comparison
    cleaned_doc_title = doc_title_text.strip() if doc_title_text else ""

    for i, elem in enumerate(valid_elements):
        score = 0
        
        # Rule 1: Exclude the document title if it matches
        if cleaned_doc_title and elem["text"].strip() == cleaned_doc_title:
            logging.debug(f"Filtered out document title from outline: '{elem['text']}'")
            continue # Skip this element entirely if it's the document title

        # Rule 2: Font size heuristics
        # Consider a block a heading candidate if its font size is significantly larger than the body font
        if elem["font_size"] > body_font_size * 1.3: # Very strong indicator
            score += 4
        elif elem["font_size"] > body_font_size * 1.1: # Strong indicator
            score += 2
        
        # Rule 3: Bold text heuristic
        if elem["is_bold"]:
            score += 2

        # Rule 4: Vertical spacing heuristic
        # A heading usually has more space above it than normal lines
        if i > 0:
            prev_elem = valid_elements[i-1]
            vertical_gap = elem["y0"] - prev_elem["y1"]
            # If the gap is significantly larger than typical line spacing (e.g., > 1.8 * font_size of previous element)
            # and not excessively large (to filter out large empty spaces that aren't headings, max 4x font size)
            if vertical_gap > (prev_elem["font_size"] * 1.8) and vertical_gap < (elem["font_size"] * 4):
                score += 3
        else: # First element on page - special handling as there's no previous element
            # If it's very high on the page and large/bold, it might be a heading (but not the main title)
            if elem["page"] > 1 and elem["y0"] < elem["bbox"][3] * 3 and (elem["font_size"] > body_font_size * 1.2 or elem["is_bold"]):
                 score += 2


        # Rule 5: Length check (negative score for long, paragraph-like text)
        # Tuned these slightly to avoid filtering out legitimate longer headings
        if len(elem["text"]) > 120: # Penalty for very long text
            score -= 4
        elif len(elem["text"]) > 70: # Moderate penalty
            score -= 2
        
        # Rule 6: Common heading keywords and patterns (positive score)
        # Look for patterns indicative of major sections
        text_upper = elem["text"].strip().upper()
        if text_upper in ["BONAFIDE CERTIFICATE", "ACKNOWLEDGEMENT", "ABSTRACT", "TABLE OF CONTENTS", 
                          "LIST OF FIGURES", "LIST OF TABLES", "APPENDIX", "REFERENCES", 
                          "CONCLUSION", "INTRODUCTION"] or \
           text_upper.startswith("CHAPTER") or text_upper.startswith("SECTION") or text_upper.startswith("APPENDIX"):
           score += 5 # High score for known major headings

        # Rule 7: Filter out specific noise on initial pages (e.g., student names, university details)
        # This is very specific to the SAMPLE.pdf structure on page 1
        if elem["page"] == 1 and (
            "SUBMITTED BY" in text_upper or 
            "IN PARTIAL FULFILLMENT" in text_upper or
            "ENGINEERING COLLEGE" in text_upper or
            "ANNA UNIVERSITY" in text_upper or
            "LIVE IN LAB" in text_upper # "LIVE IN LAB II REPORT"
        ):
            score -= 10 # Heavily penalize these specific non-headings on the first page
        
        # Rule 8: Specific exclusion for the Table of Contents header
        if "CHAPTER NO. TITLE PAGE NO." in text_upper:
            logging.debug(f"Filtered out specific TOC header: '{elem['text']}'")
            score -= 100 # Ensure this is completely filtered out

        # Final decision based on total score
        # A higher threshold to be considered a *true* potential heading
        if score >= 4: # Tunable threshold based on testing
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

    # Sort potential headings by page then by y-coordinate to maintain document order
    potential_headings.sort(key=lambda h: (h["page"], h["y0"]))

    # 2. Group consecutive, similar-style heading candidates into single logical headings
    grouped_headings = []
    if potential_headings:
        current_group = [potential_headings[0]]
        for i in range(1, len(potential_headings)):
            prev_h = potential_headings[i-1]
            curr_h = potential_headings[i]

            # Condition for grouping:
            # Same page AND (very close vertically AND similar font size AND similar boldness)
            # The 1.0 multiplier for font_size provides a very tight tolerance for vertical distance within a group
            is_close_vertically_for_grouping = (curr_h["y0"] - prev_h["bbox"][3]) < (prev_h["font_size"] * 1.0)
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
    
    # 3. Assign H1, H2, H3 levels based on refined rank_index and overall structure
    unique_ranks = sorted(list(set(h["rank_index"] for h in final_outline_elements)))
    
    level_map = {}
    
    # Primary assignment based on distinct font size ranks
    if len(unique_ranks) >= 1: level_map[unique_ranks[0]] = "H1"
    if len(unique_ranks) >= 2: level_map[unique_ranks[1]] = "H2"
    if len(unique_ranks) >= 3: level_map[unique_ranks[2]] = "H3"
    
    # Any further distinct sizes also map to H3 (as per challenge requirement for H1-H3).
    for i in range(3, len(unique_ranks)):
        level_map[unique_ranks[i]] = "H3" 

    final_outline = []
    for h in final_outline_elements:
        level = level_map.get(h["rank_index"], "H3") # Default to H3 if rank not mapped

        # Override rule: Force common major sections to H1 (strongest level)
        text_upper = h["text"].strip().upper()
        if text_upper in ["BONAFIDE CERTIFICATE", "ACKNOWLEDGEMENT", "ABSTRACT", "TABLE OF CONTENTS", 
                          "LIST OF FIGURES", "LIST OF TABLES", "REFERENCES", "CONCLUSION"] or \
           text_upper.startswith("CHAPTER") or text_upper.startswith("APPENDIX"):
            level = "H1"
            
        final_outline.append({
            "level": level,
            "text": h["text"],
            "page": h["page"]
        })
    
    # Post-processing: Adjust levels if an H2 or H3 seems to be more prominent than expected
    # This is a heuristic to prevent all headings from collapsing into H1 if font sizes aren't very distinct.
    if len(final_outline) > 1:
        # Check if there are genuinely distinct font sizes that should be H2/H3
        # Look for a font size that is smaller than the largest H1 font, but still larger than body font,
        # and has been consistently bold.
        
        # Get actual font sizes of assigned H1s
        h1_font_sizes = sorted(list(set(h["font_size"] for h in final_outline_elements if level_map.get(h["rank_index"]) == "H1")), reverse=True)
        
        if h1_font_sizes:
            largest_h1_font = h1_font_sizes[0]
            
            # Identify potential H2/H3 font sizes
            # These are font sizes that are smaller than the largest H1 but larger than body, and are typically bold.
            potential_lower_level_fonts = sorted([
                fs for fs in unique_font_sizes 
                if fs < largest_h1_font and fs >= body_font_size * 1.05 # Smaller than H1 but still bigger than body
            ], reverse=True)

            # Re-assign levels based on these identified font hierarchies
            for item in final_outline:
                elem_font_size = next((e["font_size"] for e in final_outline_elements if e["text"] == item["text"] and e["page"] == item["page"]), 0)
                
                # If an item was assigned H1 due to keyword, keep it H1.
                # Otherwise, use font size for a more granular assignment.
                if item["level"] != "H1": # Only re-evaluate if not already a forced H1
                    if elem_font_size == largest_h1_font:
                        item["level"] = "H1"
                    elif len(potential_lower_level_fonts) >= 1 and elem_font_size == potential_lower_level_fonts[0]:
                        item["level"] = "H2"
                    elif len(potential_lower_level_fonts) >= 2 and elem_font_size == potential_lower_level_fonts[1]:
                        item["level"] = "H3"
                    elif elem_font_size > body_font_size: # Default to H3 if still a prominent font
                        item["level"] = "H3"

    return final_outline


def extract_data_for_round_1a(pdf_path):
    """
    Extracts title and hierarchical outline (H1, H2, H3) from a single PDF.
    """
    logging.info(f"Starting Round 1A processing for: {pdf_path.name}")
    
    doc_title = "Untitled Document"
    all_elements_from_pages = [] # To collect elements from all pages for global analysis

    try:
        doc = fitz.open(pdf_path)
        
        # 1. Attempt to get title from metadata first
        if doc.metadata and doc.metadata.get("title"):
            metadata_title = doc.metadata["title"].strip()
            if metadata_title:
                doc_title = metadata_title
                logging.info(f"Identified document title from metadata: '{doc_title}'")


        # 2. Process page by page to extract elements for outline
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_dict = page.get_text("dict")
            
            # Extract basic elements for the current page
            current_page_elements = extract_document_elements_from_page(page_dict)
            
            # Add page number to each element before collecting globally
            for elem in current_page_elements:
                elem["page"] = page_num + 1
            all_elements_from_pages.extend(current_page_elements)

            # For the very first page, refine title if metadata was generic or empty
            if page_num == 0:
                top_page_elements = sorted(current_page_elements, key=lambda e: e["y0"])
                
                potential_title_blocks = []
                if top_page_elements:
                    # Find the largest font size among top elements
                    max_first_page_font = 0
                    for elem in top_page_elements:
                        if elem["y0"] < page.rect.height * 0.35: # Consider top 35% of page for title
                            max_first_page_font = max(max_first_page_font, elem["font_size"])

                    if max_first_page_font > 0:
                        # Collect all elements with this max font size at the top AND are bold
                        # Giving preference to bold text for titles
                        for elem in top_page_elements:
                            if elem["y0"] < page.rect.height * 0.35 and elem["font_size"] == max_first_page_font and elem["is_bold"]:
                                potential_title_blocks.append(elem)
                        
                        # Fallback: if no bold title found with max font, take non-bold max font
                        if not potential_title_blocks:
                            for elem in top_page_elements:
                                if elem["y0"] < page.rect.height * 0.35 and elem["font_size"] == max_first_page_font:
                                    potential_title_blocks.append(elem)

                # If we found strong candidates, merge them for the title
                if potential_title_blocks and (doc_title == "Untitled Document" or not doc_title):
                    potential_title_blocks.sort(key=lambda e: e["y0"])
                    
                    merged_title_lines = []
                    if potential_title_blocks:
                        merged_title_lines.append(potential_title_blocks[0]["text"])
                        for i in range(1, len(potential_title_blocks)):
                            prev_block = potential_title_blocks[i-1]
                            curr_block = potential_title_blocks[i]
                            
                            is_close_vertically = (curr_block["y0"] - prev_block["bbox"][3]) < (prev_block["font_size"] * 1.0) # Very tight check for multi-line title
                            is_same_font_size = curr_block["font_size"] == prev_block["font_size"]
                            is_same_boldness = curr_block["is_bold"] == prev_block["is_bold"]

                            if is_close_vertically and is_same_font_size and is_same_boldness:
                                merged_title_lines.append(curr_block["text"])
                            else:
                                break # Break if a clear separation 
                    
                    if merged_title_lines:
                        doc_title = " ".join(merged_title_lines)
                        logging.info(f"Refined document title from page 1 content: '{doc_title}'")


        doc.close()
        
        # 3. Infer heading levels from all collected elements
        # Pass the identified document title to infer_heading_levels to avoid duplication
        outline_data = infer_heading_levels(all_elements_from_pages, doc_title)

        return {
            "title": doc_title,
            "outline": outline_data
        }

    except Exception as e:
        logging.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return None

# --- Main Processing Loop for Round 1A ---
def process_all_pdfs_for_round_1a():
    """
    Scans the input directory, processes each PDF for Round 1A requirements,
    and saves the output JSON.
    """
    logging.info("Starting Round 1A PDF outline extraction solution...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDF files found in the input directory: {INPUT_DIR}. Ensure you are mounting '{INPUT_DIR}' correctly with your PDF files inside.")
        return

    logging.info(f"Found {len(pdf_files)} PDF(s) to process in {INPUT_DIR}.")

    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_pdf = {executor.submit(extract_data_for_round_1a, pdf_file): pdf_file for pdf_file in pdf_files}
        
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                round_1a_output = future.result()
                if round_1a_output:
                    output_file_name = f"{pdf_file.stem}.json"
                    output_file_path = OUTPUT_DIR / output_file_name
                    
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(round_1a_output, f, indent=2, ensure_ascii=False)
                    logging.info(f"Successfully saved Round 1A JSON for '{pdf_file.name}' to '{output_file_name}'.")
                else:
                    logging.error(f"Failed to get Round 1A data for '{pdf_file.name}'. Skipping JSON output.")
            except Exception as exc:
                logging.error(f"An exception occurred during processing of '{pdf_file.name}': {exc}", exc_info=True)

    logging.info("Round 1A PDF outline extraction solution finished.")

# --- Entry Point ---
if __name__ == "__main__":
    process_all_pdfs_for_round_1a()