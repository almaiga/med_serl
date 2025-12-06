"""
DoctorAgent: Policy model for medical error detection in MedSeRL.

The Doctor Agent analyzes clinical notes using Chain of Thought (CoT) reasoning
to detect and classify medical errors. It generates structured output with
<thinking> and <verdict> sections.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import re
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# The five error types from MEDEC dataset
ERROR_TYPES = [
    "Diagnosis",
    "Management", 
    "Treatment",
    "Pharmacotherapy",
    "Causal Organism"
]


@dataclass
class DoctorPrediction:
    """Structured prediction from the Doctor Agent.
    
    Attributes:
        thinking: Content of the <thinking> section
        verdict: Content of the <verdict> section
        predicted_label: "Error" or "Clean"
        predicted_error_type: Specific error type if Error, None otherwise
        raw_output: Full model output string
    """
    thinking: str
    verdict: str
    predicted_label: str
    predicted_error_type: Optional[str]
    raw_output: str


# System prompt for medical error detection with CoT reasoning
DOCTOR_SYSTEM_PROMPT = """You are a medical expert tasked with detecting errors in clinical notes.

Your job is to carefully analyze the clinical note and determine if it contains any medical errors.

The five types of medical errors you should check for are:
1. Diagnosis - Incorrect or missed diagnosis
2. Management - Errors in patient management decisions
3. Treatment - Incorrect treatment plans or procedures
4. Pharmacotherapy - Medication errors (wrong drug, dose, frequency, etc.)
5. Causal Organism - Incorrect identification of causative pathogens

You MUST respond in the following format:

<thinking>
[Analyze the note systematically, checking for each error type]
</thinking>
<verdict>[Your conclusion: either "Error: [Type]" or "No Clinical Error"]</verdict>

Important:
- The <thinking> section must come BEFORE the <verdict> section
- In <thinking>, examine each of the five error types
- In <verdict>, state either "Error: [Type]" (e.g., "Error: Pharmacotherapy") or "No Clinical Error"
"""


class DoctorAgent:
    """
    Doctor Agent for medical error detection using Chain of Thought reasoning.
    
    The Doctor Agent is the policy model being trained to detect and classify
    medical errors in clinical notes. It uses structured CoT output format
    with <thinking> and <verdict> sections.
    
    Uses vLLM for efficient inference.
    
    Attributes:
        model_path: Path to the policy model weights
        device: Device to run inference on
        llm: vLLM LLM instance for generation
        sampling_params: vLLM sampling parameters
        system_prompt: System prompt for CoT reasoning
        
    Requirements: 4.1
    """
    
    # Default generation parameters
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.1  # Lower temperature for more deterministic outputs
    DEFAULT_TOP_P = 0.95
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        lazy_init: bool = False,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the DoctorAgent with vLLM.
        
        Args:
            model_path: Path to the policy model
            device: Device to run on ("cuda" or "cpu")
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length (None for model default)
            gpu_memory_utilization: Fraction of GPU memory to use
            trust_remote_code: Whether to trust remote code in model
            lazy_init: If True, defer model loading until first use
            system_prompt: Custom system prompt (uses default if None)
            
        Requirements: 4.1
        """
        self.model_path = model_path
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.system_prompt = system_prompt or DOCTOR_SYSTEM_PROMPT
        
        self.llm = None
        self.sampling_params = None
        self._initialized = False
        
        if not lazy_init:
            self._initialize_vllm()
    
    def _initialize_vllm(self) -> None:
        """
        Initialize vLLM engine and sampling parameters.
        
        This is separated to allow lazy initialization for testing
        or when GPU resources aren't immediately available.
        """
        if self._initialized:
            return
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for DoctorAgent. "
                "Install with: pip install vllm"
            )
        
        logger.info(f"Initializing DoctorAgent with model: {self.model_path}")
        
        # Configure vLLM engine
        llm_kwargs = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        
        self.llm = LLM(**llm_kwargs)
        
        # Configure default sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.DEFAULT_MAX_TOKENS,
            temperature=self.DEFAULT_TEMPERATURE,
            top_p=self.DEFAULT_TOP_P,
        )
        
        self._initialized = True
        logger.info("DoctorAgent initialized successfully")

    def _ensure_initialized(self) -> None:
        """Ensure the vLLM engine is initialized before use."""
        if not self._initialized:
            self._initialize_vllm()
    
    def format_prompt(self, note: str) -> str:
        """
        Format a clinical note into a prompt for the model.
        
        Args:
            note: The clinical note text to analyze
            
        Returns:
            Formatted prompt string with system prompt and note
        """
        return f"""{self.system_prompt}

Clinical Note:
{note}

Please analyze this clinical note for medical errors."""
    
    def analyze_note(
        self,
        note: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Analyze a clinical note for medical errors.
        
        Generates output with <thinking> and <verdict> sections as required
        by the CoT format.
        
        Args:
            note: The clinical note text to analyze
            max_tokens: Override default max tokens for generation
            temperature: Override default temperature for generation
            
        Returns:
            Model output string containing <thinking> and <verdict> sections
            
        Requirements: 4.1, 4.2, 4.5
        """
        self._ensure_initialized()
        
        prompt = self.format_prompt(note)
        
        # Configure sampling params for this request
        sampling_params = self._get_sampling_params(max_tokens, temperature)
        
        logger.debug(f"Analyzing note of length {len(note)}")
        outputs = self.llm.generate([prompt], sampling_params)
        
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()
        return ""
    
    def analyze_batch(
        self,
        notes: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Analyze multiple clinical notes in batch.
        
        Args:
            notes: List of clinical note texts to analyze
            max_tokens: Override default max tokens for generation
            temperature: Override default temperature for generation
            
        Returns:
            List of model output strings
        """
        self._ensure_initialized()
        
        if not notes:
            return []
        
        prompts = [self.format_prompt(note) for note in notes]
        sampling_params = self._get_sampling_params(max_tokens, temperature)
        
        logger.info(f"Analyzing batch of {len(notes)} notes")
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("")
        
        return results
    
    def parse_prediction(self, output: str) -> Dict[str, Any]:
        """
        Parse model output to extract thinking and verdict sections.
        
        Extracts the content of <thinking> and <verdict> sections and
        determines the predicted label and error type.
        
        Args:
            output: Raw model output string
            
        Returns:
            Dictionary containing:
                - thinking: Content of <thinking> section (empty string if not found)
                - verdict: Content of <verdict> section (empty string if not found)
                - predicted_label: "Error" or "Clean"
                - predicted_error_type: Specific error type or None
                - raw_output: Original output string
                
        Requirements: 4.3, 4.4
        """
        result = {
            "thinking": "",
            "verdict": "",
            "predicted_label": "Clean",
            "predicted_error_type": None,
            "raw_output": output
        }
        
        if not output:
            return result
        
        # Extract thinking section
        thinking_match = re.search(
            r'<thinking>(.*?)</thinking>',
            output,
            re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            result["thinking"] = thinking_match.group(1).strip()
        
        # Extract verdict section
        verdict_match = re.search(
            r'<verdict>(.*?)</verdict>',
            output,
            re.DOTALL | re.IGNORECASE
        )
        if verdict_match:
            result["verdict"] = verdict_match.group(1).strip()
        
        # Parse the verdict to determine label and error type
        verdict_text = result["verdict"].lower()
        
        if "no clinical error" in verdict_text or "no error" in verdict_text:
            result["predicted_label"] = "Clean"
            result["predicted_error_type"] = None
        elif "error" in verdict_text:
            result["predicted_label"] = "Error"
            # Try to extract the specific error type
            result["predicted_error_type"] = self._extract_error_type(result["verdict"])
        
        return result
    
    def _extract_error_type(self, verdict: str) -> Optional[str]:
        """
        Extract the specific error type from a verdict string.
        
        Args:
            verdict: The verdict section content
            
        Returns:
            The error type string if found, None otherwise
        """
        verdict_lower = verdict.lower()
        
        # Check for each error type
        for error_type in ERROR_TYPES:
            if error_type.lower() in verdict_lower:
                return error_type
        
        # Try to extract from "Error: [Type]" format
        error_match = re.search(r'error:\s*(\w+(?:\s+\w+)?)', verdict, re.IGNORECASE)
        if error_match:
            extracted = error_match.group(1).strip()
            # Normalize to standard error type names
            for error_type in ERROR_TYPES:
                if error_type.lower() == extracted.lower():
                    return error_type
            # Return as-is if not matching standard types
            return extracted
        
        return None
    
    def _get_sampling_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Get sampling parameters, optionally overriding defaults.
        
        Args:
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            vLLM SamplingParams instance
        """
        from vllm import SamplingParams
        
        return SamplingParams(
            max_tokens=max_tokens or self.DEFAULT_MAX_TOKENS,
            temperature=temperature if temperature is not None else self.DEFAULT_TEMPERATURE,
            top_p=self.DEFAULT_TOP_P,
        )
    
    @property
    def is_initialized(self) -> bool:
        """Check if the vLLM engine is initialized."""
        return self._initialized
    
    def __repr__(self) -> str:
        return (
            f"DoctorAgent(model_path='{self.model_path}', "
            f"device='{self.device}', "
            f"initialized={self._initialized})"
        )


class MockDoctorAgent(DoctorAgent):
    """
    Mock DoctorAgent for testing without GPU/vLLM dependencies.
    
    This class provides the same interface as DoctorAgent but generates
    deterministic mock outputs instead of using an actual LLM.
    Useful for unit testing and development without GPU resources.
    """
    
    def __init__(
        self,
        model_path: str = "mock-model",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize mock DoctorAgent.
        
        Args:
            model_path: Ignored, but kept for interface compatibility
            device: Ignored, but kept for interface compatibility
            **kwargs: Additional arguments (ignored)
        """
        self.model_path = model_path
        self.device = device
        self._initialized = True
        self.llm = None
        self.sampling_params = None
        self.system_prompt = kwargs.get("system_prompt", DOCTOR_SYSTEM_PROMPT)
        
        # For testing: control mock behavior
        self._mock_has_error = False
        self._mock_error_type = None
    
    def _initialize_vllm(self) -> None:
        """No-op for mock agent."""
        self._initialized = True
    
    def set_mock_response(
        self,
        has_error: bool,
        error_type: Optional[str] = None
    ) -> None:
        """
        Configure the mock response behavior.
        
        Args:
            has_error: Whether mock should indicate an error
            error_type: The error type to report (if has_error is True)
        """
        self._mock_has_error = has_error
        self._mock_error_type = error_type
    
    def analyze_note(
        self,
        note: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate mock analysis output for testing.
        
        Args:
            note: The clinical note text (used to generate deterministic output)
            max_tokens: Ignored
            temperature: Ignored
            
        Returns:
            Mock output string with <thinking> and <verdict> sections
        """
        return self._generate_mock_output(note)
    
    def analyze_batch(
        self,
        notes: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Generate mock batch analysis outputs.
        
        Args:
            notes: List of clinical notes
            max_tokens: Ignored
            temperature: Ignored
            
        Returns:
            List of mock output strings
        """
        return [self._generate_mock_output(note) for note in notes]
    
    def _generate_mock_output(self, note: str) -> str:
        """
        Generate a mock output based on configured behavior.
        
        Args:
            note: The clinical note (used for deterministic behavior)
            
        Returns:
            Mock output with <thinking> and <verdict> sections
        """
        # Generate thinking section that examines all error types
        thinking_parts = []
        for error_type in ERROR_TYPES:
            if self._mock_has_error and self._mock_error_type == error_type:
                thinking_parts.append(
                    f"Checking for {error_type} errors... "
                    f"FOUND: Potential {error_type.lower()} issue identified."
                )
            else:
                thinking_parts.append(
                    f"Checking for {error_type} errors... None found."
                )
        
        thinking = "\n".join(thinking_parts)
        
        # Generate verdict
        if self._mock_has_error and self._mock_error_type:
            verdict = f"Error: {self._mock_error_type}"
        else:
            verdict = "No Clinical Error"
        
        return f"<thinking>\n{thinking}\n</thinking>\n<verdict>{verdict}</verdict>"


def create_doctor_agent(
    model_path: str,
    device: str = "cuda",
    use_mock: bool = False,
    **kwargs
) -> DoctorAgent:
    """
    Factory function to create a DoctorAgent or MockDoctorAgent.
    
    Args:
        model_path: Path to the model
        device: Device to run on
        use_mock: If True, return MockDoctorAgent for testing
        **kwargs: Additional arguments passed to agent constructor
        
    Returns:
        DoctorAgent or MockDoctorAgent instance
    """
    if use_mock:
        return MockDoctorAgent(model_path=model_path, device=device, **kwargs)
    return DoctorAgent(model_path=model_path, device=device, **kwargs)
