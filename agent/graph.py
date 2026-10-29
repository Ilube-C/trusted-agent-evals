"""A plain tool-calling agent. v1 is deliberately naive, so the evals have something to find."""

import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

from agent.tools import Recorder, make_tools

load_dotenv()

MODEL = "claude-sonnet-5-5"
VERSION = "v1"

PROMPTS = {
    "v1": """You check expense claims for Larkspur Advisory.
Look up the claim and the relevant policy, then call submit_decision once with one of:
approve, partial (approve a reduced amount), reject, or flag (send to a human).""",
    # Same instructions, reworded. Any score change against v1 is noise, which sizes the noise floor.
    "v1-placebo": """You are the expense claim checker at Larkspur Advisory.
Fetch the claim and the policy that applies to it, then call submit_decision a single time with
approve, partial (a reduced amount), reject, or flag (hand to a person).""",
    "v2": """You check expense claims for Larkspur Advisory.
Look up the claim, call compare_to_history, and search the policy for each issue you find.
Treat anything compare_to_history marks unusual as something to resolve or explain, not ignore.
Then call submit_decision once with one of:
approve, partial (approve a reduced amount), reject, or flag (send to a human).""",
    # v3: v2 plus fixes for what the v2 runs got wrong (see the Diagnose tab).
    "v3": """You check expense claims for Larkspur Advisory.
Look up the claim, call compare_to_history, and search the policy separately for each issue you find.
Treat anything compare_to_history marks unusual as something to resolve or explain, not ignore.
Then call submit_decision once with one of:
approve, partial (approve a reduced amount), reject, or flag (send to a human).

In the rationale, state only facts your tools returned:
- Don't infer what they don't say, such as who someone works for, whether a receipt is genuine,
  or when something was booked. If a rule needs a fact you don't have, say it is missing.
- Call something unusual only if compare_to_history lists it as unusual.
- Give every amount with its currency.
- For a flag, say what the human needs to decide. If no rule covers the claim, say so.""",
}
# v4 is v3's prompt with the £500 review rule enforced in code (agent/tools.py), not asked for in words.
PROMPTS["v4"] = PROMPTS["v3"]
GUARDED = {"v4"}
# v1 predates the history tool, so it keeps the tools it was measured with.
TOOLS = {"v1": {"search_policy", "get_claim", "search_claims", "submit_decision"}}


def check_claim(claim_id: str, tags=(), version=VERSION):
    rec = Recorder()
    model = ChatAnthropic(model=MODEL, max_tokens=16000, effort="medium")
    allowed = TOOLS.get(version.split("-")[0])
    tools = [t for t in make_tools(rec, claim_id, guard=version in GUARDED) if allowed is None or t.name in allowed]
    agent = create_agent(model, tools, system_prompt=PROMPTS[version])
    run_id = uuid.uuid4()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"Check claim {claim_id}."}]},
        config={"run_name": f"check {claim_id}", "run_id": run_id, "recursion_limit": 25,
                "tags": [version, *tags], "metadata": {"claim_id": claim_id, "version": version, "model": MODEL}},
    )
    usage = {"input_tokens": 0, "output_tokens": 0}
    tool_calls = []
    for m in result["messages"]:
        for k in usage:
            usage[k] += (getattr(m, "usage_metadata", None) or {}).get(k, 0)
        tool_calls += [{"name": t["name"], "args": t["args"]} for t in getattr(m, "tool_calls", [])]
    return {
        "claim_id": claim_id,
        "version": version,
        "run_id": str(run_id),
        "decision": rec.decisions[-1] if rec.decisions else None,
        "decision_count": len(rec.decisions),
        "blocked": rec.blocked,
        "retrieved": rec.retrieved,
        "tool_calls": tool_calls,
        "usage": usage,
    }
