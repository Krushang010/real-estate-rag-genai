# prompt.py

from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template as base_template

# --- Base LangChain-compatible prompt ---
CUSTOM_INSTRUCTION_PREFIX = """You are an expert real estate analyst. 
Use only the provided sources to answer the user's question.
If the answer isn't in the sources, say "Not found in provided sources." Do not hallucinate.
"""

CUSTOM_PROMPT_TEMPLATE = CUSTOM_INSTRUCTION_PREFIX + "\n\n" + base_template

PROMPT = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"]
)

# --- Prompt refinement logic (custom) ---
def generate_prompt(question: str, combined_text: str) -> str:
    question_lower = question.lower()

    if "challenge" in question_lower or "barrier" in question_lower:
        instruction = (
            "Extract a bullet-point list of key *regulatory and economic challenges* "
            "slowing down real estate growth in India. Avoid general advice or trends.\n\n"
        )
    elif "trend" in question_lower or "update" in question_lower:
        instruction = (
            "Summarize the following into a concise, bullet-point list of *real estate trends* "
            "in India based on the latest sources. Exclude outdated or international trends.\n\n"
        )
    elif "impact" in question_lower or "effect" in question_lower:
        instruction = (
            "Describe the *impact* or lack thereof of the following content. If no clear impact is found, list the most relevant real estate insights instead.\n\n"
        )
    elif "opportunity" in question_lower or "growth area" in question_lower:
        instruction = (
            "Extract a bullet-point list of *emerging opportunities or growth areas* in the Indian real estate market.\n\n"
        )
    else:
        instruction = (
            "Summarize the relevant insights from the content below as a concise bullet-point list.\n\n"
        )

    return instruction + combined_text
