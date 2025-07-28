import sys
import fitz  # PyMuPDF
import json
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification, AutoTokenizer
import torch
from pdf2image import convert_from_path
from collections import Counter, defaultdict
from PIL import Image

def extract_title(doc):
    # Try from PDF metadata
    title = doc.metadata.get("title", "")
    if title.strip(): return title.strip()
    # Fallback: largest text on page 1
    first_page = doc[0]
    candidates = []
    for block in first_page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                candidates.append((span["size"], span["text"].strip()))
    if not candidates: return ""
    return max(candidates, key=lambda x: x[0])[1]

def normalize_bbox(bbox, width, height):
    # Transform bbox to LayoutLM 0-1000 scale
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

def parse_pdf_with_mupdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_words = []
    page_sizes = {}
    for page_num in range(len(doc)):
        page = doc[page_num]
        width, height = page.rect.width, page.rect.height
        page_sizes[page_num] = (width, height)
        word_objs = page.get_text("words")  # list of [x0, y0, x1, y1, 'word', block_no, line_no, word_no_in_line]
        for w in word_objs:
            bbox = (w[0], w[1], w[2], w[3])
            word_text = w[4]
            all_words.append({
                "text": word_text,
                "bbox": bbox,
                "page_idx": page_num,
                "width": width,
                "height": height,
                "font_size": None # Can't get per-word font size in PyMuPDF at this level
            })
    doc.close()
    return all_words, page_sizes

def build_layoutlmv2_features(words, page_sizes, processor, max_words=512):
    images = []
    for page_idx in sorted(page_sizes.keys()):
        # Create blank image just for spatial reference (LayoutLMv2 expects image, but for text PDFs, that's ok)
        w, h = int(page_sizes[page_idx][0]), int(page_sizes[page_idx][1])
        images.append(Image.new("RGB", (w, h), "white"))

    # Chunk by page
    features = []
    for page_idx in sorted(page_sizes.keys()):
        page_words = [w for w in words if w["page_idx"] == page_idx]
        if not page_words:
            continue
        words_list = [w["text"] for w in page_words]
        boxes = [normalize_bbox(w["bbox"], w["width"], w["height"]) for w in page_words]
        feature = processor(images[page_idx], words_list, boxes=boxes, return_tensors="pt", truncation=True, max_length=max_words)
        features.append((feature, page_idx, page_words))
    return features

def label_headings_with_layoutlmv2(features, processor, model, device):
    heading_results = []
    for feature, page_idx, page_words in features:
        feature = {k: v.to(device) for k, v in feature.items()}
        outputs = model(**feature)
        predictions = torch.argmax(outputs.logits, dim=2)

        # Get label mapping
        label_map = processor.tokenizer.convert_ids_to_tokens(feature["input_ids"].squeeze().tolist())
        labels = model.config.id2label

        preds = predictions.squeeze().tolist()
        input_ids = feature["input_ids"].squeeze().tolist()
        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
        # Map tokens back to original words; processor returns offset_mapping which helps
        word_ids = feature["word_ids"].squeeze().tolist() if hasattr(feature, 'word_ids') else list(range(len(preds)))
        pred_labels = [model.config.id2label[p] for p in preds]
        # Collate predictions at word level
        word_pred = defaultdict(list)
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                word_pred[word_idx].append(pred_labels[idx])
        # For each word, majority label
        for i, wp in enumerate(page_words):
            preds_here = word_pred.get(i, [])
            if not preds_here: continue
            most_label = Counter(preds_here).most_common(1)[0][0]
            # Heuristic: treat "B-HEADER" or similar as heading (tune label according to model)
            if "HEAD" in most_label.upper():
                heading_results.append({
                    "text": wp["text"],
                    "bbox": wp["bbox"],
                    "page_idx": wp["page_idx"],
                    "font_size": wp.get("font_size", None)
                })
    return heading_results

def assign_heading_levels(headings, all_words):
    # Cluster headings using bbox['y'] (vertical position), maybe font size if available
    # Also sort headings by font size descending, assign H1/H2/H3 by uniqueness
    # If font size is None (PyMuPDF limitation), can use clustering by y-coord, length, or combine with heuristics

    # Heuristic fallback: rank headings by their heights (upper y=smaller value)
    all_headings = []
    for h in headings:
        all_headings.append(
            (h["page_idx"], h["text"], h["bbox"], h.get("font_size"))
        )
    if not all_headings:
        return []

    # For better accuracy: try to get font sizes from blocks that match the heading text and bbox
    font_sizes = []
    for h in all_headings:
        _, text, bbox, _ = h
        word_candidates = [w for w in all_words if w["text"].strip().lower() == text.strip().lower() and abs(w["bbox"][1] - bbox[1]) < 5]
        if word_candidates:
            size = word_candidates[0].get("font_size")
            if size: font_sizes.append(size)
    # If font sizes available, cluster them
    level_map = {}
    if font_sizes:
        uniq_sizes = sorted(list(set(font_sizes)), reverse=True)
        for i, sz in enumerate(uniq_sizes):
            level_map[sz] = f"H{i+1}" if i < 6 else f"H6"
    else:
        # Fallback: assign by Y-position (upper lines = H1)
        levels = ["H1", "H2", "H3", "H4", "H5", "H6"]
        y_coords = sorted(list(set([h[2][1] for h in all_headings])))
        for i, yc in enumerate(y_coords):
            level_map[yc] = levels[i] if i < 6 else "H6"
    res = []
    for h in all_headings:
        page, text, bbox, size = h
        key = size if font_sizes else bbox[1]
        level = level_map.get(key, "H3")
        res.append({
          "level": level,
          "text": text,
          "page": page + 1,
        })
    return res

def main(pdf_path, output_json="output.json"):
    # 1. Extract raw text and layout from PDF
    words, page_sizes = parse_pdf_with_mupdf(pdf_path)
    # 2. Load LayoutLMv2 processor/model
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
    model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 3. LayoutLMv2 feature extraction
    features = build_layoutlmv2_features(words, page_sizes, processor)
    # 4. Label headings
    headings = label_headings_with_layoutlmv2(features, processor, model, device)
    # 5. Assign heading levels (heuristic or hybrid)
    outline = assign_heading_levels(headings, words)
    # 6. Extract title
    with fitz.open(pdf_path) as doc:
        title = extract_title(doc)
    result = {
        "title": title,
        "outline": outline
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Extracted outline written to {output_json}")

if _name_ == "_main_":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Input PDF file")
    parser.add_argument("--output", default="output.json", help="Output JSON file")
    args = parser.parse_args()
    main(args.pdf_path, args.output)
