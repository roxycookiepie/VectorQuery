import os
from typing import Optional
import json
import logging
from typing import Dict, Any, Tuple, List
from API_Azure import AzureOpenAIPlatformService


class ProposalIngestor:
    def __init__(self):
        self.prompt = """
            You are analyzing a civil engineering proposal or RFQ. Produce one output containing two main sections in this exact order:
 
            ---
            
            Section 1: Summary
            
            Write a factual, professional summary of 150–250 words (extend up to 400 for complex proposals).
            The summary must describe the project being proposed, focusing on scope, scale, and technical purpose — not firm qualifications.
            
            Include only information explicitly stated in the document.
            Do not infer, generalize, or fabricate any details.
            Do not include employee names or resumes in this section.
            
            Emphasize:
            - Project title, client/owner, location (city/state/region), and completion date (if provided)
            - Primary purpose and type of project (e.g., water line, roadway, drainage, site development, wastewater treatment)
            - Detailed scope elements, including explicit measurements or quantities such as:
              - Linear feet of pipe/utilities
              - Pipe diameters
              - Roadway miles or lane miles
              - Treatment plant capacity (MGD)
              - Detention/retention volume (CY)
              - Construction cost or budget
            - Disciplines and service types (e.g., civil, structural, surveying, environmental, construction management)
            - Unique challenges, constraints, or proposed solutions relevant to the scope
            
            Keep the focus on what is being designed or constructed, not who is doing it.
            
            ---
            
            Section 2: Metadata
            
            Provide a structured metadata list with one entry per field:
            
            Employees:
            [ { "name": "Full Name, Credentials", "role": "Position Title" }, ... ]
            
            Client/Owner: [name]
            
            Project Title: [title]
            
            Project Location: [city, county, and/or state]
            
            Proposal Type: [e.g., RFQ, RFP, SOQ]
            
            Disciplines Involved: [civil, structural, survey, environmental, etc.]
            
            Project Type / Keywords: [e.g., "large diameter water line", "storm drain improvements"]
            
            Key Quantities: [e.g., "5 miles of 36-inch water line"]
            
            Completion Date: [date or "Indefinite Deliverable Contract"]
            
            Estimated Construction Cost: [amount or "N/A"]
            
            Notable Challenges or Constraints: [brief description]
            
            Other Notable Metadata: [any other relevant information]
            
            ---
            
            IMPORTANT OUTPUT FORMAT:
            Return your response as a single JSON object with one key "chunk_text" containing the entire formatted output.
            
            The chunk_text should contain both Section 1 and Section 2 as formatted text (not nested JSON).
            
            Example structure:
            {
              "chunk_text": "Section 1: Summary\\n\\n[Your summary here...]\\n\\nSection 2: Metadata\\n\\nEmployees:\\n[...]\\n\\nClient/Owner: [...]\\n\\nProject Title: [...]\\n..."
            }
            
            If any metadata field is missing, write "Unknown."
            Never fabricate or infer missing data.
            
            Purpose: Your output will be vectorized as-is to enable semantic search queries like
            "return large diameter water line proposals that Mike Fitzgerald was included on in Houston"
            """
        self._openai = AzureOpenAIPlatformService()

    def process_single_proposal(self, file_name: str, full_text: str) -> Tuple[List[Tuple[Optional[int], int, str]], Dict[str, Any]]:
        """
        Process proposal and return a single chunk tuple with combined summary + metadata.
        
        Returns:
            Tuple of (chunk_tuples, empty_dict) where chunk_tuples contains ONE item: [(None, 0, combined_text)]
        """
        logging.info(f"Starting proposal processing for: {file_name}, text length: {len(full_text)} chars")
        
        if not full_text or not full_text.strip():
            logging.error(f"Empty text provided for {file_name}")
            return [], {}
        
        try:
            combined_chunk_text = self._analyze_with_llm(file_name, full_text)
            
            if not combined_chunk_text or len(combined_chunk_text.strip()) < 50:
                logging.error(f"LLM returned empty or insufficient content for {file_name}")
                return [], {}
            
            logging.info(f"LLM generated combined chunk: {len(combined_chunk_text)} chars")
            
        except Exception as e:
            logging.error(f"Proposal processing failed for {file_name}: {e}", exc_info=True)
            return [], {}

        # Return single chunk tuple: (page_num=None, chunk_idx=0, chunk_text)
        chunk_tuples = [(None, 0, combined_chunk_text)]
        
        logging.info(f"Processed proposal {file_name}: 1 combined chunk generated")
        
        # Return empty dict since metadata is already embedded in chunk_text
        return chunk_tuples, {}

    def _analyze_with_llm(self, file_name: str, full_text: str) -> str:
        """Send extracted text to Azure OpenAI for analysis and return combined chunk_text."""
        MAX_CHARS = 120000
        if len(full_text) > MAX_CHARS:
            logging.warning(f"Truncating proposal text from {len(full_text)} to {MAX_CHARS} chars")
            full_text = full_text[:MAX_CHARS]
        
        user_message = f"Please analyze this proposal document:\n\n{full_text}"
        
        try:
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_message}
            ]
            response = self._openai.chat_completion(messages=messages, temperature=0.3)
            
            logging.info(f"LLM response received for {file_name}: {len(response)} chars")
            
            # Parse the JSON response to extract chunk_text
            chunk_text = self._parse_llm_response(response)
            return chunk_text
            
        except Exception as e:
            logging.error(f"LLM analysis failed for {file_name}: {e}", exc_info=True)
            return ""

    def _parse_llm_response(self, response_text: str) -> str:
        """Parse LLM response to extract the chunk_text field."""
        try:
            # Try to parse as JSON first
            response_text = response_text.strip()
            
            # Handle code block wrapper if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
            
            response_text = response_text.strip()
            
            # Parse JSON
            parsed = json.loads(response_text)
            
            if isinstance(parsed, dict) and "chunk_text" in parsed:
                chunk_text = parsed["chunk_text"]
                logging.info(f"Successfully extracted chunk_text: {len(chunk_text)} chars")
                return chunk_text
            else:
                logging.warning("Response JSON missing 'chunk_text' field, using raw response")
                return response_text
                
        except json.JSONDecodeError as je:
            logging.warning(f"Failed to parse JSON response: {je}. Using raw text as chunk_text.")
            return response_text
        except Exception as e:
            logging.error(f"Unexpected error parsing LLM response: {e}", exc_info=True)
            return response_text
