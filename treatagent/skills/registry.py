from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SkillSpec:
    name: str
    description: str
    evidence_category: str
    triggers: List[str] = field(default_factory=list)
    input_schema: List[str] = field(default_factory=list)
    output_schema: List[str] = field(default_factory=list)
    cost: float = 1.0
    local_only: bool = True
    supports_cutoff_filtering: bool = False
    executor: Optional[Callable[[str, str], Dict[str, Any]]] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "evidence_category": self.evidence_category,
            "triggers": self.triggers,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "cost": self.cost,
            "local_only": self.local_only,
            "supports_cutoff_filtering": self.supports_cutoff_filtering,
        }


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillSpec] = {}

    def register(self, skill: SkillSpec) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillSpec:
        return self._skills[name]

    def names(self) -> List[str]:
        return list(self._skills.keys())

    def metadata(self) -> List[Dict[str, Any]]:
        return [self._skills[name].to_metadata() for name in self.names()]

    def execute(self, name: str, smiles: str, disease: str) -> Dict[str, Any]:
        skill = self.get(name)
        if skill.executor is None:
            raise ValueError(f"Skill {name} does not have an executor.")
        return skill.executor(smiles, disease)
