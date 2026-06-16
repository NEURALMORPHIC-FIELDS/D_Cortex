# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# The professional control organism and the deterministic baseline. answer() is the
# SINGLE public path to a user-facing answer; it always returns the output of the
# hard verifier. No code path returns an answer that has not passed the verifier, so
# a final answer cannot reach the user without passing D_Cortex.

from dataclasses import dataclass, field
from typing import List, Optional

from dcortex_professional.pack import ProfessionalPack
from dcortex_professional.enforcement import (DomainRouter, MemoryStateResolver, HardVerifier,
                                              Resolution, Claim, COMMITTED, PROVISIONAL, DISPUTED,
                                              UNKNOWN, OUT_OF_DOMAIN, FORBIDDEN)
from dcortex_professional.runtime import SubstrateLM


@dataclass
class Answer:
    text: str
    state: str
    action: str                       # answer / abstain / uncertain / provisional / block / out_of_domain
    grounded: bool
    claims: List[Claim] = field(default_factory=list)
    source_path: str = "none"         # deterministic_lookup / neural_binder / verifier / mixed / none
    emission_path: str = "none"       # constrained_decode / template / abstain_template / none
    verifier_passed: bool = True
    veto_reason: str = ""
    unconstrained_slot: str = ""      # raw-model slot emission (evidence the constraint mattered)
    overridden: bool = False


def _abstain(message: str) -> str:
    return f"[ABSTAIN] {message}"


class DCortexProfessionalControl:
    """Mechanical grounding organism. answer() is the only choke point."""

    def __init__(self, pack: ProfessionalPack, lm: Optional[SubstrateLM] = None) -> None:
        self.pack = pack
        self.router = DomainRouter(pack)
        self.resolver = MemoryStateResolver(pack)
        self.verifier = HardVerifier(pack)
        self.lm = lm

    # ---- the single public entry point ----
    def answer(self, query: str) -> Answer:
        in_domain, risk = self.router.route(query)
        resolution = self.resolver.resolve(query, in_domain, risk)
        text, claims, emission_path, ev = self._generate(query, resolution)
        return self._finalize(query, resolution, text, claims, emission_path, ev)

    # ---- generation under constraint (never returned directly to the user) ----
    def _generate(self, query: str, r: Resolution):
        ev = {"unconstrained_slot": "", "overridden": False}
        if r.state == OUT_OF_DOMAIN:
            return ("[OUT-OF-DOMAIN] This query is outside the patent-analysis domain.",
                    [], "abstain_template", ev)
        if r.state == UNKNOWN:
            return (_abstain("Not grounded in D_Cortex_PatentAnalyst committed memory."),
                    [], "abstain_template", ev)
        if r.state == DISPUTED:
            cands = "; ".join(f"{c['value']} ({c['source']})" for c in r.disputed["candidates"])
            claim = Claim(r.entity, r.attribute, r.disputed["candidates"][0]["value"], DISPUTED)
            return (f"[UNCERTAIN] {r.entity} {r.attribute}: conflicting sources -> {cands}.",
                    [claim], "template", ev)
        if r.state == PROVISIONAL:
            claim = Claim(r.entity, r.attribute, r.value, PROVISIONAL)
            return (f"[PROVISIONAL] {r.entity} {r.attribute} is {r.value} (not independently verified).",
                    [claim], "template", ev)
        if r.state == COMMITTED:
            claim = Claim(r.entity, r.attribute, r.value, COMMITTED)
            if self.lm is not None and self.lm.available:
                prompt = f"{r.entity} {r.attribute} is"
                cr = self.lm.generate_constrained(prompt, r.value)
                ev = {"unconstrained_slot": cr.unconstrained_slot_text, "overridden": cr.overridden}
                return (cr.text + ".", [claim], "constrained_decode", ev)
            return (f"{r.entity} {r.attribute} is {r.value}.", [claim], "template", ev)
        return (_abstain("Unresolved memory state."), [], "abstain_template", ev)

    # ---- the verifier veto: the single return path ----
    def _finalize(self, query: str, r: Resolution, text: str, claims: List[Claim],
                  emission_path: str, ev: dict) -> Answer:
        check = self.verifier.check(query, text, claims, r)
        if check.passed:
            action = {COMMITTED: "answer", PROVISIONAL: "provisional", DISPUTED: "uncertain",
                      UNKNOWN: "abstain", OUT_OF_DOMAIN: "out_of_domain"}.get(r.state, "abstain")
            grounded = r.state in (COMMITTED,)
            if r.state == COMMITTED:
                source = "mixed" if emission_path == "constrained_decode" else "deterministic_lookup"
            elif r.state in (PROVISIONAL, DISPUTED):
                source = "deterministic_lookup"
            else:
                source = "verifier"
            return Answer(text=text, state=r.state, action=action, grounded=grounded, claims=claims,
                          source_path=source, emission_path=emission_path, verifier_passed=True,
                          unconstrained_slot=ev.get("unconstrained_slot", ""),
                          overridden=ev.get("overridden", False))
        # VETO: a candidate asserted an ungrounded / forbidden / contaminated claim.
        if check.forbidden_hit is not None:
            blocked = f"[BLOCKED] {check.forbidden_hit['reason']}"
            return Answer(text=blocked, state=FORBIDDEN, action="block", grounded=False, claims=[],
                          source_path="verifier", emission_path="abstain_template",
                          verifier_passed=False, veto_reason=check.reason)
        # tighten: force abstain and re-verify (an empty-claim abstain always passes)
        forced = _abstain("Candidate failed verification; refusing to assert an ungrounded claim.")
        recheck = self.verifier.check(query, forced, [], r)
        return Answer(text=forced, state=UNKNOWN, action="abstain", grounded=False, claims=[],
                      source_path="verifier", emission_path="abstain_template",
                      verifier_passed=recheck.passed, veto_reason=check.reason)

    # ---- adversarial probe used by the unbypassability gate ----
    def rogue_then_verify(self, query: str, rogue_text: str, rogue_claims: List[Claim]) -> Answer:
        """Inject a rogue (ungrounded) candidate and route it through finalize, proving
        the verifier vetoes it. Used by G7/G8; not a user-facing path."""
        in_domain, risk = self.router.route(query)
        r = self.resolver.resolve(query, in_domain, risk)
        return self._finalize(query, r, rogue_text, rogue_claims, "rogue", {})


class DeterministicBaseline:
    """Pure lookup + verifier, no neural model. The honest floor for comparison."""

    def __init__(self, pack: ProfessionalPack) -> None:
        self.pack = pack
        self.router = DomainRouter(pack)
        self.resolver = MemoryStateResolver(pack)
        self.verifier = HardVerifier(pack)

    def answer(self, query: str) -> Answer:
        in_domain, risk = self.router.route(query)
        r = self.resolver.resolve(query, in_domain, risk)
        if r.state == OUT_OF_DOMAIN:
            text, claims = "[OUT-OF-DOMAIN] Outside the patent-analysis domain.", []
        elif r.state == COMMITTED:
            text, claims = f"{r.entity} {r.attribute} is {r.value}.", [Claim(r.entity, r.attribute, r.value, COMMITTED)]
        elif r.state == PROVISIONAL:
            text, claims = f"[PROVISIONAL] {r.entity} {r.attribute} is {r.value}.", [Claim(r.entity, r.attribute, r.value, PROVISIONAL)]
        elif r.state == DISPUTED:
            cands = "; ".join(f"{c['value']} ({c['source']})" for c in r.disputed["candidates"])
            text, claims = f"[UNCERTAIN] {r.entity} {r.attribute}: {cands}.", [Claim(r.entity, r.attribute, r.disputed["candidates"][0]["value"], DISPUTED)]
        else:
            text, claims = _abstain("Not in committed memory."), []
        check = self.verifier.check(query, text, claims, r)
        if not check.passed:
            return Answer(text=_abstain("Vetoed."), state=UNKNOWN, action="abstain", grounded=False,
                          source_path="verifier", emission_path="abstain_template", verifier_passed=False,
                          veto_reason=check.reason)
        action = {COMMITTED: "answer", PROVISIONAL: "provisional", DISPUTED: "uncertain",
                  OUT_OF_DOMAIN: "out_of_domain", UNKNOWN: "abstain"}.get(r.state, "abstain")
        return Answer(text=text, state=r.state, action=action, grounded=(r.state == COMMITTED),
                      claims=claims, source_path="deterministic_lookup", emission_path="template",
                      verifier_passed=True)
