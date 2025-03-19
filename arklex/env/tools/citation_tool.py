import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from arklex.env.tools.tools import Tool, register_tool
from arklex.utils.model_config import MODEL
from arklex.utils.model_provider_config import PROVIDER_MAP


logger = logging.getLogger(__name__)


@register_tool(
    desc="Generate proper citations for evidence and sources",
    slots=[
        {
            "name": "argument",
            "type": "string",
            "description": "The argument to generate citations for",
            "prompt": "Please provide the argument you need citations for.",
            "required": True
        },
        {
            "name": "topic",
            "type": "string",
            "description": "The topic of the argument",
            "prompt": "What is the topic of this argument?",
            "required": True
        }
    ],
    outputs=[
        {
            "name": "citations",
            "type": "list",
            "description": "List of generated citations"
        }
    ]
)
class CitationTool(Tool):
    """Tool for generating citations and references."""
    
    def __init__(self):
        super().__init__(
            func=self._generate_citations,
            name="CitationTool",
            description="Generate proper citations for evidence and sources",
            slots=[
                {
                    "name": "argument",
                    "type": "string",
                    "description": "The argument to generate citations for",
                    "prompt": "Please provide the argument you need citations for.",
                    "required": True
                },
                {
                    "name": "topic",
                    "type": "string",
                    "description": "The topic of the argument",
                    "prompt": "What is the topic of this argument?",
                    "required": True
                }
            ],
            outputs=[
                {
                    "name": "citations",
                    "type": "list",
                    "description": "List of generated citations"
                }
            ],
            isComplete=lambda x: True,
            isResponse=False
        )
        self.formats = ["APA", "MLA", "Chicago"]
        self.include_links = True
        self.citation_types = {
            "academic": {
                "description": "Academic papers and research",
                "examples": ["Journal articles", "Conference papers", "Theses"]
            },
            "news": {
                "description": "News articles and media",
                "examples": ["Newspaper articles", "Online news", "Magazines"]
            },
            "expert": {
                "description": "Expert opinions and interviews",
                "examples": ["Expert interviews", "Professional opinions", "Industry reports"]
            },
            "statistic": {
                "description": "Statistical data and studies",
                "examples": ["Research statistics", "Survey data", "Statistical reports"]
            },
            "study": {
                "description": "Scientific studies and research",
                "examples": ["Scientific papers", "Research studies", "Clinical trials"]
            }
        }
        self.persuasion_citation_mapping = {
            "logos": ["academic", "statistic", "study"],
            "ethos": ["expert", "academic", "news"],
            "pathos": ["news", "expert"]
        }
        self.llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(
            model=MODEL["model_type_or_path"], timeout=30000
        )
        
        self.citation_prompt = PromptTemplate.from_template(
            """Generate citations for the following argument.
            
            Argument: {argument}
            Persuasion Type: {persuasion_type}
            Citation Types: {citation_types}
            
            Available Citation Types:
            {available_types}
            
            Respond in the following JSON format:
            {{
                "citations": [
                    {{
                        "type": "citation type",
                        "content": "citation text",
                        "format": "citation format",
                        "relevance_score": float (0-1)
                    }}
                ],
                "summary": "brief summary of how citations support the argument"
            }}
            
            Citations:"""
        )

    def _generate_citations(self, argument: str, topic: str) -> Dict[str, Any]:
        """Generate citations for the given argument."""
        try:
            # Format the prompt with available citation types and formats
            citation_types_str = "\n".join([
                f"- {type_name}:\n  Description: {info['description']}\n  Examples: {', '.join(info['examples'])}"
                for type_name, info in self.citation_types.items()
            ])
            
            formats_str = ", ".join(self.formats)
            prompt = f"""Generate citations for the following argument:

Argument: {argument}
Topic: {topic}

Available Citation Types:
{citation_types_str}

Supported Formats: {formats_str}
Include Links: {self.include_links}

Generate 2-3 relevant citations that support this argument. For each citation:
1. Choose an appropriate citation type
2. Generate a citation in one of the supported formats
3. Include a link if possible
4. Provide a brief explanation of how it supports the argument

Output Format:
{{
    "citations": [
        {{
            "type": "citation_type",
            "format": "format_name",
            "citation": "formatted_citation",
            "link": "url_if_available",
            "support": "brief_explanation"
        }}
    ]
}}"""

            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Parse and validate the response
            citations = self._parse_response(response)
            self._validate_citations(citations)
            
            return {
                "success": True,
                "citations": citations
            }
            
        except Exception as e:
            logger.error(f"Error generating citations: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "citations": []
            }

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response into a structured format."""
        try:
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            if "citations" not in data:
                raise ValueError("No citations found in response")
                
            return data["citations"]
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise

    def _validate_citations(self, citations: List[Dict[str, Any]]) -> None:
        """Validate the citation structure and values."""
        required_fields = ["type", "format", "citation", "support"]
        
        for citation in citations:
            # Check required fields
            for field in required_fields:
                if field not in citation:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Validate citation type
            if citation["type"] not in self.citation_types:
                raise ValueError(f"Invalid citation type: {citation['type']}")
                
            # Validate format
            if citation["format"] not in self.formats:
                raise ValueError(f"Invalid format: {citation['format']}")
                
            # Validate link if present
            if "link" in citation and not citation["link"].startswith(("http://", "https://")):
                raise ValueError(f"Invalid link format: {citation['link']}")

    def _validate_citations(self, citations: Dict[str, Any]) -> bool:
        """Validates the citations output format."""
        required_fields = ["citations", "summary"]
        
        # Check required fields
        if not all(field in citations for field in required_fields):
            logger.error("Missing required fields in citations")
            return False
            
        # Validate each citation
        for citation in citations["citations"]:
            required_citation_fields = [
                "type", "content", "format", "relevance_score"
            ]
            
            if not all(field in citation for field in required_citation_fields):
                logger.error("Missing required fields in citation")
                return False
                
            # Validate citation type
            if citation["type"] not in self.citation_types:
                logger.error(f"Invalid citation type: {citation['type']}")
                return False
                
            # Validate relevance score
            if not 0 <= citation["relevance_score"] <= 1:
                logger.error("Invalid relevance score in citation")
                return False
                
            # Validate formatted citations
            if not all(format in self.formats for format in citation["format"]):
                logger.error("Missing required citation formats")
                return False
                
            # Validate URL if include_links is True
            if self.include_links and not citation.get("url"):
                logger.error("Missing URL when include_links is True")
                return False
                
        return True

    def generate_citations(self, argument: str, persuasion_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generates citations to support the argument.
        
        Args:
            argument: The argument to generate citations for
            persuasion_type: Optional persuasion type to filter citation types
            
        Returns:
            Dictionary containing citations and summary, or None if generation fails
        """
        try:
            # Get citation types based on persuasion type if provided
            citation_types = None
            if persuasion_type and persuasion_type in self.persuasion_citation_mapping:
                citation_types = self.persuasion_citation_mapping[persuasion_type]
            else:
                citation_types = list(self.citation_types.keys())
                
            # Format citation types for prompt
            citation_types_str = "\n".join(
                f"- {type_name}: {info['description']}\n"
                f"  Examples: {', '.join(info['examples'])}"
                for type_name, info in self.citation_types.items()
                if type_name in citation_types
            )
            
            # Format the prompt
            formatted_prompt = self.citation_prompt.format(
                argument=argument,
                persuasion_type=persuasion_type,
                citation_types=citation_types_str,
                available_types=citation_types_str
            )
            
            # Generate citations
            chain = self.llm | StrOutputParser()
            citations_str = chain.invoke(formatted_prompt)
            
            # Parse the citations
            citations = json.loads(citations_str)
            
            # Validate the citations
            if not self._validate_citations(citations):
                return None
                
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {str(e)}")
            return None

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the citation tool.
        
        Args:
            **kwargs: Must include:
                - argument: The argument to generate citations for
                - persuasion_type: Optional persuasion type to filter citation types
                
        Returns:
            Dictionary containing citations and summary
        """
        # Extract required arguments
        argument = kwargs.get("argument")
        persuasion_type = kwargs.get("persuasion_type")
        
        if not argument:
            logger.error("Missing required argument")
            return {
                "citations": [],
                "summary": "Missing required argument"
            }
            
        # Generate citations
        citations = self.generate_citations(argument, persuasion_type)
        
        if not citations:
            return {
                "citations": [],
                "summary": "Failed to generate citations"
            }
            
        return citations 