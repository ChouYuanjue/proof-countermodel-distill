from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


VARIABLE_TOKENS = {"something", "someone"}
LITERAL_RE = re.compile(r'\("([^"]+)" "([^"]+)" "([^"]+)" "([+\-~])"\)')
CHAIN_TOKEN_RE = re.compile(r"(triple\d+|rule\d+)")
NEGATED_RELATION = {
    "chases": "chase",
    "eats": "eat",
    "likes": "like",
    "needs": "need",
    "sees": "see",
    "visits": "visit",
}


def _normalize_sign(sign: str) -> str:
    return "-" if sign in {"-", "~"} else "+"


def unique_in_order(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class Literal:
    subject: str
    relation: str
    object: str
    sign: str

    def normalized(self) -> "Literal":
        return Literal(
            subject=self.subject,
            relation=self.relation,
            object=self.object,
            sign=_normalize_sign(self.sign),
        )

    def negate(self) -> "Literal":
        return Literal(
            subject=self.subject,
            relation=self.relation,
            object=self.object,
            sign="-" if self.normalized().sign == "+" else "+",
        )

    def to_text(self) -> str:
        if self.relation == "is":
            return (
                f"{self.subject} is {self.object}."
                if self.sign == "+"
                else f"{self.subject} is not {self.object}."
            )
        negated_relation = NEGATED_RELATION.get(self.relation, self.relation)
        return (
            f"{self.subject} {self.relation} {self.object}."
            if self.sign == "+"
            else f"{self.subject} does not {negated_relation} {self.object}."
        )


@dataclass(frozen=True)
class Rule:
    rule_id: str
    antecedents: tuple[Literal, ...]
    consequent: Literal
    text: str


@dataclass(frozen=True)
class Theory:
    facts: dict[str, Literal]
    rules: dict[str, Rule]


@dataclass(frozen=True)
class Derivation:
    literal: Literal
    chain_tokens: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class Failure:
    target: Literal
    reason: str
    rule_id: str | None = None
    children: tuple["Failure", ...] = ()


def parse_literal_repr(rep: str) -> Literal:
    match = LITERAL_RE.fullmatch(rep.strip())
    if match is None:
        raise ValueError(f"Could not parse literal representation: {rep}")
    subject, relation, obj, sign = match.groups()
    return Literal(subject=subject, relation=relation, object=obj, sign=_normalize_sign(sign))


def parse_rule_repr(rule_id: str, text: str, rep: str) -> Rule:
    parts = [Literal(*groups[:3], _normalize_sign(groups[3])) for groups in LITERAL_RE.findall(rep)]
    if len(parts) < 2:
        raise ValueError(f"Could not parse rule representation: {rep}")
    return Rule(rule_id=rule_id, antecedents=tuple(parts[:-1]), consequent=parts[-1], text=text)


def extract_chain_tokens(question: dict) -> tuple[str, ...]:
    proofs = question.get("proofsWithIntermediates")
    if proofs:
        representation = proofs[0]["representation"]
        return unique_in_order(CHAIN_TOKEN_RE.findall(representation))
    proof_text = question.get("proofs", "")
    return unique_in_order(CHAIN_TOKEN_RE.findall(proof_text))


def build_theory(raw_example: dict) -> Theory:
    facts: dict[str, Literal] = {}
    rules: dict[str, Rule] = {}

    for fact_id, payload in raw_example["triples"].items():
        if payload is None:
            continue
        facts[fact_id] = parse_literal_repr(payload["representation"])

    for rule_id, payload in raw_example["rules"].items():
        if payload is None:
            continue
        rules[rule_id] = parse_rule_repr(
            rule_id=rule_id,
            text=payload["text"],
            rep=payload["representation"],
        )

    return Theory(facts=facts, rules=rules)


def _is_variable(token: str) -> bool:
    return token in VARIABLE_TOKENS


def _extend_binding(binding: dict[str, str], pattern: str, actual: str) -> dict[str, str] | None:
    if not _is_variable(pattern):
        return binding if pattern == actual else None
    current = binding.get(pattern)
    if current is None:
        new_binding = dict(binding)
        new_binding[pattern] = actual
        return new_binding
    return binding if current == actual else None


def unify(pattern: Literal, literal: Literal, binding: dict[str, str] | None = None) -> dict[str, str] | None:
    if pattern.sign != literal.sign:
        return None
    binding = {} if binding is None else dict(binding)
    for pattern_part, actual_part in (
        (pattern.subject, literal.subject),
        (pattern.relation, literal.relation),
        (pattern.object, literal.object),
    ):
        binding = _extend_binding(binding, pattern_part, actual_part)
        if binding is None:
            return None
    return binding


def instantiate(pattern: Literal, binding: dict[str, str]) -> Literal:
    def resolve(token: str) -> str:
        return binding.get(token, token)

    return Literal(
        subject=resolve(pattern.subject),
        relation=resolve(pattern.relation),
        object=resolve(pattern.object),
        sign=pattern.sign,
    )


def _match_rule(
    rule: Rule,
    derivations: dict[Literal, Derivation],
    index: int = 0,
    binding: dict[str, str] | None = None,
    premises: tuple[Derivation, ...] = (),
):
    if index == len(rule.antecedents):
        yield (binding or {}, premises)
        return

    antecedent = rule.antecedents[index]
    for literal, derivation in derivations.items():
        new_binding = unify(antecedent, literal, binding)
        if new_binding is None:
            continue
        yield from _match_rule(
            rule=rule,
            derivations=derivations,
            index=index + 1,
            binding=new_binding,
            premises=premises + (derivation,),
        )


def forward_chain(theory: Theory, max_steps: int = 64) -> dict[Literal, Derivation]:
    derivations: dict[Literal, Derivation] = {
        literal: Derivation(literal=literal, chain_tokens=(fact_id,), source=fact_id)
        for fact_id, literal in theory.facts.items()
    }

    for _ in range(max_steps):
        changed = False
        current = dict(derivations)
        for rule in theory.rules.values():
            for binding, premises in _match_rule(rule, current):
                conclusion = instantiate(rule.consequent, binding)
                candidate_chain = unique_in_order(
                    [token for premise in premises for token in premise.chain_tokens] + [rule.rule_id]
                )
                existing = derivations.get(conclusion)
                if existing is None or len(candidate_chain) < len(existing.chain_tokens):
                    derivations[conclusion] = Derivation(
                        literal=conclusion,
                        chain_tokens=candidate_chain,
                        source=rule.rule_id,
                    )
                    changed = True
        if not changed:
            break
    return derivations


def _consequent_bindings(rule: Rule, target: Literal) -> list[dict[str, str]]:
    binding = unify(rule.consequent, target)
    return [] if binding is None else [binding]


def _failure_score(failure: Failure) -> tuple[int, int, str]:
    leaves = collect_missing_literals(failure)
    return (len(leaves), failure_depth(failure), failure.rule_id or "")


def failure_depth(failure: Failure) -> int:
    if not failure.children:
        return 1
    return 1 + max(failure_depth(child) for child in failure.children)


def collect_missing_literals(failure: Failure) -> tuple[Literal, ...]:
    if not failure.children:
        return (failure.target,)
    items: list[Literal] = []
    for child in failure.children:
        items.extend(collect_missing_literals(child))
    return tuple(unique_literal_order(items))


def unique_literal_order(items: Iterable[Literal]) -> list[Literal]:
    seen: set[Literal] = set()
    ordered: list[Literal] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def explain_failure(
    target: Literal,
    theory: Theory,
    derivations: dict[Literal, Derivation],
    max_depth: int = 3,
    visited: frozenset[Literal] | None = None,
) -> Failure:
    if target in derivations:
        return Failure(target=target, reason="proved")

    visited = frozenset() if visited is None else visited
    if target in visited:
        return Failure(target=target, reason="cycle")

    candidates: list[Failure] = []
    for rule in theory.rules.values():
        for binding in _consequent_bindings(rule, target):
            children: list[Failure] = []
            for antecedent in rule.antecedents:
                grounded = instantiate(antecedent, binding)
                if grounded in derivations:
                    continue
                if max_depth <= 0:
                    children.append(Failure(target=grounded, reason="missing"))
                else:
                    children.append(
                        explain_failure(
                            target=grounded,
                            theory=theory,
                            derivations=derivations,
                            max_depth=max_depth - 1,
                            visited=visited | {target},
                        )
                    )
            if children:
                candidates.append(
                    Failure(
                        target=target,
                        reason="rule_failure",
                        rule_id=rule.rule_id,
                        children=tuple(children),
                    )
                )
    if not candidates:
        return Failure(target=target, reason="no_rule")
    return min(candidates, key=_failure_score)


def failure_chain_tag(failure: Failure) -> str:
    if failure.reason == "no_rule":
        return "FAIL[no_rule]"
    if failure.reason == "cycle":
        return "FAIL[cycle]"
    if failure.rule_id:
        return f"FAIL[{failure.rule_id}]"
    return "FAIL[missing]"


def failure_witness_text(failure: Failure) -> str:
    if failure.reason == "no_rule":
        return f"No rule or fact supports: {failure.target.to_text()}"
    if failure.reason == "cycle":
        return f"Cyclic dependency while trying to derive: {failure.target.to_text()}"
    missing = collect_missing_literals(failure)
    missing_text = "; ".join(literal.to_text() for literal in missing)
    if failure.rule_id:
        return f"{failure.rule_id} is blocked by missing support: {missing_text}"
    return missing_text


def verify_chain(
    theory: Theory,
    chain_tokens: Iterable[str],
    target: Literal,
) -> bool:
    available: dict[Literal, str] = {}

    for token in chain_tokens:
        if token in theory.facts:
            available[theory.facts[token]] = token
            continue
        rule = theory.rules.get(token)
        if rule is None:
            return False
        matched = False
        for binding, _ in _match_rule(rule, {lit: Derivation(lit, (src,), src) for lit, src in available.items()}):
            conclusion = instantiate(rule.consequent, binding)
            available[conclusion] = token
            matched = True
        if not matched:
            return False

    return target in available
