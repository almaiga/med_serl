# MedSeRL Agents Module
# Contains Scribe Agent and Doctor Agent implementations

from .scribe_agent import ScribeAgent, MockScribeAgent, create_scribe_agent
from .doctor_agent import (
    DoctorAgent,
    MockDoctorAgent,
    create_doctor_agent,
    DoctorPrediction,
    ERROR_TYPES,
    DOCTOR_SYSTEM_PROMPT
)

__all__ = [
    "ScribeAgent",
    "MockScribeAgent",
    "create_scribe_agent",
    "DoctorAgent",
    "MockDoctorAgent",
    "create_doctor_agent",
    "DoctorPrediction",
    "ERROR_TYPES",
    "DOCTOR_SYSTEM_PROMPT"
]
