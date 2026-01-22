"""Custom reward function for MedSeRL self-play training.

Following verl documentation format:
https://verl.readthedocs.io/en/latest/preparation/reward_function.html
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward score for medical error detection self-play.
    
    For now, this is a placeholder that returns a simple reward.
    In full self-play implementation, this will evaluate game outcomes.
    
    Args:
        data_source (str): Dataset identifier (e.g., 'medec_selfplay')
        solution_str (str): The model's generated response
        ground_truth (str): Ground truth note_id from data
        extra_info (dict): Additional information from dataset
        
    Returns:
        float: Reward score (0.0 to 1.0)
    """
    # For now, return neutral reward to allow training to proceed
    # TODO: Implement proper game-based reward evaluation
    return 0.5
