from dataclasses import dataclass


@dataclass
class LabelAndProb:
    label: str
    prob: float

@dataclass
class EmissionResult:
    model_name: str

    step1_emission: float
    step2_emission: float
    step3_emission: float
    step4_emission: float

    step1_time: float
    step2_time: float
    step3_time: float
    step4_time: float

    inference_result: list[dict]
