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

Your task is to create an EXTREMELY DETAILED and COMPREHENSIVE guide that covers:

1. Code Organization and Structure:
   - Directory structure best practices for {library_name}
   - File naming conventions specific to {library_name}
   - Module organization best practices for projects using {library_name}
   - Component architecture recommendations for {library_name}
   - Code splitting strategies appropriate for {library_name}

2. Common Patterns and Anti-patterns:
   - Design patterns specific to {library_name}
   - Recommended approaches for common tasks with {library_name}
   - Anti-patterns and code smells to avoid when using {library_name}
   - State management best practices for {library_name} applications
   - Error handling patterns appropriate for {library_name}

3. Performance Considerations:
   - Optimization techniques specific to {library_name}
   - Memory management considerations for applications using {library_name}
   - Rendering optimization for {library_name} (if applicable)
   - Bundle size optimization strategies for projects using {library_name}
   - Lazy loading strategies appropriate for {library_name}

4. Security Best Practices:
   - Common vulnerabilities and how to prevent them with {library_name}
   - Input validation best practices for {library_name}
   - Authentication and authorization patterns for {library_name}
   - Data protection strategies relevant to {library_name}
   - Secure API communication with {library_name}

5. Testing Approaches:
   - Unit testing strategies for {library_name} components
   - Integration testing approaches for {library_name} applications
   - End-to-end testing recommendations for {library_name} projects
   - Test organization best practices for {library_name}
   - Mocking and stubbing techniques specific to {library_name}

6. Common Pitfalls and Gotchas:
   - Frequent mistakes developers make when using {library_name}
   - Edge cases to be aware of when using {library_name}
   - Version-specific issues with {library_name}
   - Compatibility concerns between {library_name} and other technologies
   - Debugging strategies for {library_name} applications

7. Tooling and Environment:
   - Recommended development tools for {library_name}
   - Build configuration best practices for projects using {library_name}
   - Linting and formatting recommendations for {library_name} code
   - Deployment best practices for {library_name} applications
   - CI/CD integration strategies for {library_name} projects

Format your response as a valid JSON object with exactly these keys:
  - name: a short descriptive name for the rule (e.g., "{library_name} Best Practices")
  - glob_pattern: the most appropriate glob pattern for this library based on the file types it typically works with
  - description: a clear 1-2 sentence description of what the rule covers
  - content: the formatted rule content with comprehensive best practices in markdown format
"""

        # Determine if we need to generate content from scratch
        if len(exa_answer.strip()) < 100 and len(all_citation_text.strip()) < 200:
            # Not enough content from Exa, generate from scratch
            exa_content_section = f"""I need you to research and generate comprehensive best practices for {library_info.name} from your knowledge.

Please be extremely thorough and detailed, covering all aspects of {library_info.name} development.
Your guidance should be useful for both beginners and experienced developers.
"""

            prompt = enhanced_prompt_template.format(
                library_name=library_info.name,
                tags=tags_str,
                exa_content_section=exa_content_section,
                mdc_instructions=mdc_instructions,
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
