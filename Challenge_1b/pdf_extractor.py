import fitz  # PyMuPDF library
import logging
from pathlib import Path
import json # Import json for table content serialization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_elements_from_page_dict(page_dict, page_obj):
    """
    Analyzes a page dictionary from PyMuPDF to infer structure,
    including headings, paragraphs, and tables, returning raw elements.
    Ensures all returned elements have consistent keys.
    """
    elements = [] # Will store (sort_key, element_data) tuples

    # Store bounding boxes of detected tables to exclude their text from general text processing
    table_bboxes = []
    
    # 1. Detect Tables First and extract their content
    tables = page_obj.find_tables()
    for table in tables:
        table_bbox = table.bbox
        # Ensure table_bbox is valid before processing
        if not (isinstance(table_bbox, (list, tuple)) and len(table_bbox) == 4):
            logging.warning(f"Malformed table bbox: {table_bbox} on page {page_obj.number + 1}. Skipping table.")
            continue
        
        table_bboxes.append(fitz.Rect(table_bbox))
        
        # Extract headers and rows as text directly using table.extract()
        table_content_data = []
        try:
            extracted_cells = table.extract() 
            if extracted_cells:
                table_content_data = [list(row) for row in extracted_cells] # Ensure it's list of lists for JSON
        except Exception as e:
            logging.warning(f"Error extracting table content using table.extract() for bbox {table_bbox}: {e}")
            table_content_data = [] # Fallback to empty if extraction fails

        # Create a text representation of the table for general processing (e.g., semantic analysis)
        table_text_representation = ""
        if table_content_data:
            for row in table_content_data:
                table_text_representation += " ".join(str(cell) for cell in row if cell is not None) + "\n"
            table_text_representation = table_text_representation.strip()

        table_element_data = {
            "type": "table",
            "content": table_content_data, # Actual structured table data
            "bbox": list(table_bbox),
            "font_size": 0,       # Tables don't have a direct font size, use 0 or avg body font size if possible
            "is_bold": False,     # Tables are not inherently bold
            "y0": table_bbox[1],  # Y-coordinate for sorting
            "y1": table_bbox[3],  # Y-coordinate for span calculation
            "text": table_text_representation # Text representation of the table for semantic search
        }
        elements.append((table_bbox[1], table_element_data)) # Use y0 for sorting

    # 2. Process Text Blocks (excluding areas covered by tables)
    blocks = []
    for block in page_dict.get("blocks", []):
        if block["type"] == 0:  # text block
            block_bbox = fitz.Rect(block["bbox"])
            
            is_part_of_table = False
            for t_bbox in table_bboxes:
                if block_bbox.intersects(t_bbox) or t_bbox.contains(block_bbox):
                    is_part_of_table = True
                    break
            
            if not is_part_of_table:
                blocks.append(block)

    # Sort non-table blocks by their top-left Y-coordinate, then by X-coordinate
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    # Collect all font sizes and their counts from the remaining (non-table) text
    font_sizes = {}
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = round(span["size"], 1)
                font_sizes[size] = font_sizes.get(size, 0) + len(span["text"])

    body_font_size = 0
    heading_candidates_sizes = set()

    if font_sizes:
        sorted_font_sizes = sorted(font_sizes.items(), key=lambda item: item[1], reverse=True)
        body_font_size = sorted_font_sizes[0][0] if sorted_font_sizes else 0 

        for size, char_count in sorted_font_sizes:
            if size > body_font_size * 1.2: 
                heading_candidates_sizes.add(size)
        
        if not heading_candidates_sizes and sorted_font_sizes: 
            heading_candidates_sizes.add(sorted_font_sizes[0][0])


    page_height = page_dict.get("height", 0)
    TOP_MARGIN_THRESHOLD = page_height * 0.1
    BOTTOM_MARGIN_THRESHOLD = page_height * 0.9

    for block in blocks:
        block_text_lines = []
        block_font_size = 0 
        is_bold = False
        
        for line in block.get("lines", []):
            line_text_parts = []
            for span in line.get("spans", []):
                line_text_parts.append(span["text"])
                if not block_font_size:
                    block_font_size = round(span["size"], 1)
                if span["flags"] & 2: # Bit flag for bold
                    is_bold = True
            block_text_lines.append(" ".join(line_text_parts).strip())
        
        cleaned_block_text = " ".join(block_text_lines).replace(" -\n", "").replace("-\n", "").strip()

        if not cleaned_block_text:
            continue

        x0, y0, x1, y1 = block["bbox"]
        is_numeric_like = cleaned_block_text.strip().replace('-', '').replace('.', '').replace(' ', '').isdigit()
        is_in_top_margin = y0 < TOP_MARGIN_THRESHOLD
        is_in_bottom_margin = y1 > BOTTOM_MARGIN_THRESHOLD
        
        if is_numeric_like and (is_in_top_margin or is_in_bottom_margin) and len(cleaned_block_text.strip()) <= 5:
            continue

        element_type = "paragraph"
        if block_font_size in heading_candidates_sizes and len(cleaned_block_text) < 200:
            element_type = "heading"
        
        elements.append((y0, {
            "type": element_type, 
            "content": cleaned_block_text, # Main content, for text blocks this is 'text'
            "bbox": list(block["bbox"]),
            "font_size": block_font_size, 
            "is_bold": is_bold,           
            "y0": y0,                     
            "y1": y1,                     
            "text": cleaned_block_text # Redundant for text, but ensures consistency for sorting/filtering
        }))

    elements.sort(key=lambda item: item[0])
    
    return [item[1] for item in elements]

def extract_raw_data_from_pdf(pdf_path):
    """
    Extracts raw structured data (elements per page) from a single PDF using PyMuPDF.
    Returns a dictionary with all_elements (flat list), doc_title, and page_count.
    """
    logging.info(f"Starting raw extraction for PDF: {pdf_path.name}")
    
    all_elements = []
    doc_title = "Untitled Document" 
    page_count = 0 

    try:
        doc = fitz.open(pdf_path)
        
        doc_title = (doc.metadata.get("title", "Untitled Document") or "Untitled Document").strip()
        page_count = doc.page_count 

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_dict = page.get_text("dict")

            current_page_elements = extract_elements_from_page_dict(page_dict, page)
            for elem in current_page_elements:
                elem["page"] = page_num + 1 # Add page number to each element directly
            all_elements.extend(current_page_elements) # Extend with the list of elements from this page

        doc.close() 

        logging.info(f"Finished raw extraction for: {pdf_path.name}")
        return {
            "all_elements": all_elements,
            "doc_title": doc_title,
            "page_count": page_count 
        }

    except Exception as e:
        logging.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return None

