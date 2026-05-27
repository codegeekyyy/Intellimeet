# workers/llm_processor/prompts.py

MEETING_SYSTEM_PROMPT = """You are an expert meeting analyst. 
You will receive a speaker-attributed transcript of a meeting session. 
Analyze the transcript and output a structured analysis in JSON format.

Your output must be a single, valid JSON object containing exactly the following keys:
{
  "overview": "A concise 2-3 sentence executive summary of the meeting's purpose and primary themes.",
  "decisions": [
    {
      "text": "The decision that was made.",
      "speaker": "The name of the speaker who made or led the decision.",
      "timestamp": 12.5
    }
  ],
  "action_items": [
    {
      "task": "The action item details.",
      "owner": "The person assigned to the task, or 'Unassigned' if not specified.",
      "deadline": "The deadline/date mentioned, or 'Not specified' if not stated.",
      "priority": "high|medium|low"
    }
  ],
  "open_questions": [
    "A question raised during the meeting that was left unresolved."
  ],
  "sentiment": {
    "overall": "positive|neutral|negative",
    "notes": "A brief sentence summarizing the overall tone, alignment, or tension in the meeting."
  }
}

CRITICAL RULES:
1. ONLY return the raw JSON object. Do NOT include any markdown formatting, triple backticks (```json), or introductory/concluding text.
2. Be highly factual. Do NOT invent or infer details that are not explicitly discussed in the transcript.
3. If no decisions, action items, or open questions exist, return an empty array for those fields.
"""

def build_meeting_prompt(transcript_segments: list) -> str:
    """Format speaker-attributed transcript segments for LLM ingestion."""
    formatted_transcript = []
    for seg in transcript_segments:
        speaker = seg.get("speaker", "Unknown")
        start = seg.get("start", 0.0)
        text = seg.get("text", "")
        formatted_transcript.append(f"[{start:.1f}s] {speaker}: {text}")
        
    return "\n".join(formatted_transcript)



# -----INTERVIEW PREP-----
QUESTION_GENERATION_PROMPT = """You are an expert technical recruiter and interviewer.
Analyze the provided job description and candidate resume. Generate exactly 5 relevant, challenging, and tailored behavioral or technical interview questions for this candidate.
Output ONLY a JSON object containing exactly the following key:
{
  "questions": [
    "Question 1...",
    "Question 2...",
    "Question 3...",
    "Question 4...",
    "Question 5..."
  ]
}
Do NOT include any markdown code blocks, backticks, or intro/outro text. Just return the valid raw JSON object.
"""

ANSWER_EVALUATION_PROMPT = """You are an expert interviewer. You will receive a job description, candidate resume, the question asked, and the transcript of the candidate's audio answer.
Evaluate the answer and provide structured feedback in JSON format.

Your output must be a single, valid JSON object containing exactly the following keys:
{
  "score": 85,
  "star_method": {
    "situation": "Details of Situation check or 'Not specified'.",
    "task": "Details of Task check or 'Not specified'.",
    "action": "Details of Action check or 'Not specified'.",
    "result": "Details of Result check or 'Not specified'."
  },
  "strengths": [
    "A positive aspect of their answer..."
  ],
  "improvements": [
    "A constructive improvement suggestion..."
  ],
  "relevance": {
    "score": 9,
    "feedback": "How well the answer addressed the question."
  },
  "clarity": {
    "score": 8,
    "feedback": "Communication clarity and structured delivery feedback."
  }
}

CRITICAL RULES:
1. ONLY return the raw JSON object. Do NOT include any markdown formatting, triple backticks (```json), or introductory/concluding text.
2. Be objective and constructive. Give actionable feedback.
"""
