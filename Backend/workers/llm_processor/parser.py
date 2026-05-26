# workers/llm_processor/parser.py

import re
import json
import logging

logger = logging.getLogger("llm_worker.parser")

def parse_summary_response(raw_text: str) -> dict:
    """
    Cleans markdown code fences, extracts JSON substring,
    parses it, and validates/fills the required keys.
    """
    cleaned = raw_text.strip()
    
    # 1. Strip triple backticks and markdown code block indicators
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    
    # 2. Extract JSON using regex search in case LLM added extra text
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed standard JSON decode. Attempting regex extract...")
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Regex JSON extraction failed: {e}")
                raise ValueError("Response contains invalid JSON structure") from e
        else:
            raise ValueError("No JSON object structure found in response")

    # 3. Validate and enforce schema constraints
    defaults = {
        "overview": "No overview summary provided.",
        "decisions": [],
        "action_items": [],
        "open_questions": [],
        "sentiment": {"overall": "neutral", "notes": "No sentiment analysis notes provided."}
    }
    
    validated = {}
    for key, default in defaults.items():
        val = data.get(key)
        
        # Verify lists
        if isinstance(default, list):
            if isinstance(val, list):
                validated[key] = val
            else:
                validated[key] = []
        # Verify dict (sentiment)
        elif isinstance(default, dict):
            if isinstance(val, dict):
                inner_validated = {}
                for k, v in default.items():
                    inner_validated[k] = val.get(k, v)
                validated[key] = inner_validated
            else:
                validated[key] = default
        # Verify strings
        else:
            validated[key] = str(val) if val is not None else default
            
    return validated