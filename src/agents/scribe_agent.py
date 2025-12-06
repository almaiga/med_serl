"""
ScribeAgent: Data generation component for MedSeRL training.

Uses MedGemma-4B (or compatible model) via vLLM to execute text transformations
according to the 4-Quadrant Strategy for generating diverse training data.

Requirements: 3.1, 3.6
"""

from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ScribeAgent:
    """
    Scribe Agent for generating training data via text transformations.
    
    The Scribe Agent uses a medical language model (MedGemma-4B) to transform
    clinical notes according to four strategies:
    - Augmented Ground Truth: Preserve errors while changing demographics
    - Augmented Safe: Paraphrase while maintaining medical accuracy
    - Synthetic Decoy: Inject cosmetic noise without medical errors
    - Synthetic Injection: Inject new medical errors
    
    Uses vLLM for efficient batch inference.
    
    Attributes:
        model_path: Path to the model weights
        device: Device to run inference on
        llm: vLLM LLM instance for generation
        sampling_params: vLLM sampling parameters
    """
    
    # Default generation parameters
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        lazy_init: bool = False
    ):
        """
        Initialize the ScribeAgent with vLLM.
        
        Args:
            model_path: Path to the MedGemma-4B model or compatible model
            device: Device to run on ("cuda" or "cpu")
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length (None for model default)
            gpu_memory_utilization: Fraction of GPU memory to use
            trust_remote_code: Whether to trust remote code in model
            lazy_init: If True, defer model loading until first use
            
        Requirements: 3.1
        """
        self.model_path = model_path
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        
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
                "vLLM is required for ScribeAgent. "
                "Install with: pip install vllm"
            )
        
        logger.info(f"Initializing ScribeAgent with model: {self.model_path}")
        
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
        logger.info("ScribeAgent initialized successfully")

    def _ensure_initialized(self) -> None:
        """Ensure the vLLM engine is initialized before use."""
        if not self._initialized:
            self._initialize_vllm()
    
    def transform_batch(
        self,
        prompts: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[Dict]:
        """
        Execute transformation prompts using the LLM and attach ground truth metadata.
        
        Takes a batch of transformation prompts (from MedicalDataProcessor.get_quadrant_batch)
        and generates transformed clinical notes using the LLM.
        
        Args:
            prompts: List of prompt dictionaries from get_quadrant_batch, each containing:
                - scribe_prompt: The transformation instruction
                - meta: Ground truth metadata (has_error, error_type, source)
                - original_text: Original clinical note
                - mode: Transformation mode
            max_tokens: Override default max tokens for generation
            temperature: Override default temperature for generation
            
        Returns:
            List of result dictionaries, each containing:
                - transformed_text: The LLM-generated transformed note
                - ground_truth: Ground truth metadata with has_error field
                - original_text: Original clinical note
                - mode: Transformation mode used
                - scribe_prompt: The prompt used for transformation
                
        Requirements: 3.1, 3.6
        """
        self._ensure_initialized()
        
        if not prompts:
            return []
        
        # Extract just the prompt strings for batch generation
        prompt_strings = [p["scribe_prompt"] for p in prompts]
        
        # Configure sampling params for this batch
        sampling_params = self._get_sampling_params(max_tokens, temperature)
        
        # Generate transformations in batch
        logger.info(f"Generating {len(prompt_strings)} transformations")
        outputs = self.llm.generate(prompt_strings, sampling_params)
        
        # Combine outputs with metadata
        results = []
        for prompt_dict, output in zip(prompts, outputs):
            # Extract generated text from vLLM output
            generated_text = output.outputs[0].text if output.outputs else ""
            
            # Build result with ground truth metadata attached
            result = {
                "transformed_text": generated_text.strip(),
                "ground_truth": self._build_ground_truth(prompt_dict["meta"]),
                "original_text": prompt_dict["original_text"],
                "mode": prompt_dict["mode"],
                "scribe_prompt": prompt_dict["scribe_prompt"]
            }
            results.append(result)
        
        logger.info(f"Generated {len(results)} transformed notes")
        return results
    
    def _build_ground_truth(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ground truth metadata from prompt metadata.
        
        Ensures the ground truth has the correct has_error field based on
        the transformation mode/source.
        
        Args:
            meta: Metadata from the prompt containing source and error info
            
        Returns:
            Ground truth dictionary with has_error correctly set
            
        Requirements: 3.6
        """
        ground_truth = {
            "has_error": meta.get("has_error", False),
            "source": meta.get("source", "unknown")
        }
        
        # Include error_type if present (for error samples)
        if meta.get("error_type"):
            ground_truth["error_type"] = meta["error_type"]
        
        return ground_truth
    
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
            temperature=temperature or self.DEFAULT_TEMPERATURE,
            top_p=self.DEFAULT_TOP_P,
        )
    
    def _apply_transformation(self, prompt: str, mode: str) -> str:
        """
        Apply a single transformation using the LLM.
        
        This is a convenience method for single-prompt transformations.
        For efficiency, prefer transform_batch for multiple prompts.
        
        Args:
            prompt: The transformation prompt
            mode: The transformation mode (for logging)
            
        Returns:
            Generated transformed text
        """
        self._ensure_initialized()
        
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()
        return ""
    
    def _validate_transformation(
        self,
        original: str,
        transformed: str,
        mode: str
    ) -> bool:
        """
        Validate that a transformation maintains required properties.
        
        This is a basic validation that checks the transformation produced
        meaningful output. More sophisticated validation would require
        medical NLP tools.
        
        Args:
            original: Original clinical note
            transformed: Transformed clinical note
            mode: Transformation mode
            
        Returns:
            True if transformation appears valid, False otherwise
        """
        # Basic validation: transformed text should not be empty
        if not transformed or not transformed.strip():
            logger.warning(f"Empty transformation for mode {mode}")
            return False
        
        # Transformed should be different from original (except for edge cases)
        if transformed.strip() == original.strip():
            logger.warning(f"Transformation identical to original for mode {mode}")
            return False
        
        # Transformed should have reasonable length
        if len(transformed) < 10:
            logger.warning(f"Transformation too short for mode {mode}: {len(transformed)} chars")
            return False
        
        return True
    
    @property
    def is_initialized(self) -> bool:
        """Check if the vLLM engine is initialized."""
        return self._initialized
    
    def __repr__(self) -> str:
        return (
            f"ScribeAgent(model_path='{self.model_path}', "
            f"device='{self.device}', "
            f"initialized={self._initialized})"
        )



class MockScribeAgent(ScribeAgent):
    """
    Mock ScribeAgent for testing without GPU/vLLM dependencies.
    
    This class provides the same interface as ScribeAgent but generates
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
        Initialize mock ScribeAgent.
        
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
    
    def _initialize_vllm(self) -> None:
        """No-op for mock agent."""
        self._initialized = True
    
    def transform_batch(
        self,
        prompts: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[Dict]:
        """
        Generate mock transformations for testing.
        
        Creates deterministic mock outputs based on the transformation mode.
        
        Args:
            prompts: List of prompt dictionaries
            max_tokens: Ignored
            temperature: Ignored
            
        Returns:
            List of mock result dictionaries with ground truth attached
        """
        if not prompts:
            return []
        
        results = []
        for prompt_dict in prompts:
            mode = prompt_dict.get("mode", "unknown")
            original = prompt_dict.get("original_text", "")
            meta = prompt_dict.get("meta", {})
            
            # Generate mock transformed text based on mode
            transformed = self._generate_mock_text(original, mode, meta)
            
            result = {
                "transformed_text": transformed,
                "ground_truth": self._build_ground_truth(meta),
                "original_text": original,
                "mode": mode,
                "scribe_prompt": prompt_dict.get("scribe_prompt", "")
            }
            results.append(result)
        
        return results
    
    def _generate_mock_text(
        self,
        original: str,
        mode: str,
        meta: Dict[str, Any]
    ) -> str:
        """
        Generate mock transformed text based on mode.
        
        Args:
            original: Original clinical note
            mode: Transformation mode
            meta: Metadata including error type
            
        Returns:
            Mock transformed text
        """
        if mode == "augment_error":
            error_type = meta.get("error_type", "medical")
            return f"[MOCK AUGMENTED ERROR - {error_type}] {original[:100]}..."
        
        elif mode == "augment_safe":
            return f"[MOCK PARAPHRASED] {original[:100]}..."
        
        elif mode == "make_decoy":
            return f"[MOCK DECOY WITH TYPOS] {original[:100]}..."
        
        elif mode == "inject_new_error":
            error_type = meta.get("error_type", "medical")
            return f"[MOCK INJECTED {error_type} ERROR] {original[:100]}..."
        
        else:
            return f"[MOCK TRANSFORMED] {original[:100]}..."
    
    def _apply_transformation(self, prompt: str, mode: str) -> str:
        """Generate mock single transformation."""
        return f"[MOCK {mode.upper()}] Transformed clinical note."


# Factory function for creating appropriate agent
def create_scribe_agent(
    model_path: str,
    device: str = "cuda",
    use_mock: bool = False,
    **kwargs
) -> ScribeAgent:
    """
    Factory function to create a ScribeAgent or MockScribeAgent.
    
    Args:
        model_path: Path to the model
        device: Device to run on
        use_mock: If True, return MockScribeAgent for testing
        **kwargs: Additional arguments passed to agent constructor
        
    Returns:
        ScribeAgent or MockScribeAgent instance
    """
    if use_mock:
        return MockScribeAgent(model_path=model_path, device=device, **kwargs)
    return ScribeAgent(model_path=model_path, device=device, **kwargs)
