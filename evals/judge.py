"""LLM judge for the parts of the ladder rules can't settle: leftover facts (rung 3) and whether the
rationale would satisfy a reviewer (rung 4). It sees what the agent saw, and no answer key."""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from evals.graders import observations

load_dotenv()

JUDGE_MODEL = "claude-haiku-5-5"
PRICES = (0.10, 0.50)  # $ per million input, output tokens

PROMPT = """You review the work of an automated expense-claim checker at a small consultancy.
Below is everything its tools returned during this run, then its decision. Judge the rationale
only against this evidence.

{evidence}

Decision: {decision}, approved amount {amount}, citing {cited}
Rationale: {rationale}

Answer:
- supported: every statement in the rationale is backed by the evidence above. Reasonable
  arithmetic on amounts in the evidence counts as backed.
- explained: a finance reviewer could act on this rationale without re-reading the claim: it names
  the facts that drove the decision and the rule each one meets or breaks.
- reason: one sentence on the weakest point."""


class Verdict(BaseModel):
    supported: bool
    explained: bool
    reason: str = Field(description="One sentence on the weakest point.")


VOTES = 3  # Haiku can't be pinned to temperature 0 here, so one call flips on borderline rationales


def judge(result, case):
    """Ask the judge VOTES times and take the majority on each criterion."""
    d = result["decision"]
    evidence = "\n\n".join(f"{name}({args}) returned:\n{out}" for name, args, out in observations(result))
    model = ChatAnthropic(model=JUDGE_MODEL, max_tokens=2000).with_structured_output(
        Verdict, method="json_schema", include_raw=True)
    prompt = PROMPT.format(evidence=evidence, decision=d["decision"], amount=d["approved_amount"],
                           cited=", ".join(d["clause_ids"]) or "nothing", rationale=d["rationale"])
    outs = model.batch([prompt] * VOTES, config={"run_name": f"judge {result['claim_id']}", "tags": ["judge"]})
    votes = [o["parsed"].model_dump() for o in outs]
    v = {k: sum(x[k] for x in votes) > VOTES / 2 for k in ("supported", "explained")}
    # Give the reason from a voter that agrees with the majority on both criteria, if there is one.
    agree = [x for x in votes if x["supported"] == v["supported"] and x["explained"] == v["explained"]]
    v["reason"] = (agree or votes)[0]["reason"]
    v["split"] = [k for k in ("supported", "explained") if len({x[k] for x in votes}) > 1]
    v["unanimous"] = not v["split"]
    v["votes"] = [{k: x[k] for k in ("supported", "explained")} for x in votes]
    v["cost"] = sum(o["raw"].usage_metadata["input_tokens"] / 1e6 * PRICES[0]
                    + o["raw"].usage_metadata["output_tokens"] / 1e6 * PRICES[1] for o in outs)
    return v
