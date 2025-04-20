import json
from typing import Any, Dict, List

from litellm import completion
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential

from config import CONFIG
from logger import logger
from models import LibraryInfo, MDCRule


@sleep_and_retry
@limits(calls=CONFIG.api.rate_limit_calls, period=CONFIG.api.rate_limit_period)
@retry(
    stop=stop_after_attempt(CONFIG.api.max_retries),
    wait=wait_exponential(
        multiplier=1, min=CONFIG.api.retry_min_wait, max=CONFIG.api.retry_max_wait
    ),
)
def generate_mdc_rules_from_exa(
    library_info: LibraryInfo, exa_results: Dict[str, Any]
) -> List[MDCRule]:
    """Generate MDC rules for a library using LLM directly from Exa results."""
    try:
        # Load MDC instructions
        mdc_instructions_path = CONFIG.paths.mdc_instructions
        if not mdc_instructions_path.exists():
            logger.warning(
                f"MDC instructions file not found at {mdc_instructions_path}"
            )
            mdc_instructions = (
                "Create rules with clear descriptions and appropriate glob patterns."
            )
        else:
            try:
                mdc_instructions = mdc_instructions_path.read_text()
            except Exception as e:
                logger.warning(f"Could not read MDC instructions file: {str(e)}")
                mdc_instructions = "Create rules with clear descriptions and appropriate glob patterns."

        # Load LLM instructions
        llm_instructions_path = CONFIG.paths.llm_instructions
        if not llm_instructions_path.exists():
            logger.warning(
                f"LLM instructions file not found at {llm_instructions_path}"
            )
            llm_instructions = (
                "Create rules with clear descriptions and appropriate glob patterns."
            )
        else:
            try:
                llm_instructions = llm_instructions_path.read_text()
            except Exception as e:
                logger.warning(f"Could not read LLM instructions file: {str(e)}")
                llm_instructions = "Create rules with clear descriptions and appropriate glob patterns."

        # Load LLM instructions failover
        failover_exa_instructions_path = CONFIG.paths.failover_exa_instructions
        if not failover_exa_instructions_path.exists():
            logger.warning(
                f"LLM instructions failover file not found at {failover_exa_instructions_path}"
            )
            failover_exa_instructions = (
                "Create rules with clear descriptions and appropriate glob patterns."
            )
        else:
            try:
                failover_exa_instructions = failover_exa_instructions_path.read_text()
            except Exception as e:
                logger.warning(
                    f"Could not read LLM instructions failover file: {str(e)}"
                )
                failover_exa_instructions = "Create rules with clear descriptions and appropriate glob patterns."

        # Extract Exa answer and citations
        exa_answer = exa_results.get("answer", "")
        citations = exa_results.get("citations", [])

        # Extract text from citations
        citation_texts = []
        for citation in citations:
            if isinstance(citation, dict) and citation.get("text"):
                citation_texts.append(citation.get("text", ""))
            elif hasattr(citation, "text") and citation.text:
                citation_texts.append(citation.text)

        # Combine all citation texts - Gemini can handle much larger inputs
        all_citation_text = "\n\n".join(citation_texts)

        # Format tags for prompt - but don't emphasize them too much
        tags_str = ", ".join(library_info.tags)

        # Enhanced prompt template for both cases
        enhanced_prompt_template = """Create a comprehensive Cursor rule file (.mdc) for the {library_name} library following these guidelines:

{mdc_instructions}

Library Information:
- Name: {library_name}
- Tags: {tags}

{exa_content_section}

{llm_instructions}

Format your response as a valid JSON object with exactly these keys:
  - name: a short descriptive name for the rule (e.g., "{library_name} Best Practices")
  - glob_pattern: the most appropriate glob pattern for this library based on the file types it typically works with
  - description: a clear 1-2 sentence description of what the rule covers
  - content: the formatted rule content with comprehensive best practices in markdown format
"""

        # Determine if we need to generate content from scratch
        if len(exa_answer.strip()) < 100 and len(all_citation_text.strip()) < 200:
            # Not enough content from Exa, generate from scratch
            exa_content_section = f"""{llm_instructions}"""

            prompt = enhanced_prompt_template.format(
                library_name=library_info.name,
                tags=tags_str,
                exa_content_section=exa_content_section,
                mdc_instructions=mdc_instructions,
                llm_instructions=failover_exa_instructions,
            )
        else:
            # Use existing Exa content
            chunk_size = CONFIG.processing.chunk_size
            exa_content_section = f"""Based on the following information about {library_info.name} best practices:

Exa search results:
{exa_answer}

Additional information from citations:
{all_citation_text[:chunk_size]}

Please synthesize, enhance, and expand upon this information to create the most comprehensive guide possible.
Add any important best practices that might be missing from the search results.
"""

            prompt = enhanced_prompt_template.format(
                library_name=library_info.name,
                tags=tags_str,
                exa_content_section=exa_content_section,
                mdc_instructions=mdc_instructions,
                llm_instructions=llm_instructions,
            )

        logger.info(f"Sending enhanced prompt to LLM for {library_info.name}")
        response = completion(
            model=CONFIG.api.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short descriptive name for the rule",
                        },
                        "glob_pattern": {
                            "type": "string",
                            "description": "Valid glob pattern for target files",
                        },
                        "description": {
                            "type": "string",
                            "description": "1-2 sentence description of what the rule does",
                        },
                        "content": {
                            "type": "string",
                            "description": "Formatted rule content using markdown",
                        },
                    },
                    "required": ["name", "glob_pattern", "description", "content"],
                },
            },
        )

        # Parse the JSON response
        json_response = json.loads(response.choices[0].message.content)
        logger.info(f"Successfully generated enhanced rule for {library_info.name}")

        # Create MDCRule instance from the JSON
        rule = MDCRule(**json_response)
        return [rule]

    except Exception as e:
        logger.error(f"Error generating MDC rule for {library_info.name}: {str(e)}")
        raise
