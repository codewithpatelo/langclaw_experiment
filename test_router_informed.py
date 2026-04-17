"""Smoke test for the LangGraphInformedRouter (Phase 1.3).

Validates:
  1. The informed router can be instantiated and its enum mode is recognised.
  2. The structural-feature builder (`SotopiaEnvironment._compute_router_features`)
     produces the expected schema per agent.
  3. The selection function picks a valid agent ID even when the LLM call is
     mocked (no real API contact).

Run:  python test_router_informed.py
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

from langclaw.router_informed import LangGraphInformedRouter
from langclaw.simulation import OrchestrationMode, SotopiaEnvironment


def _expected_features_schema() -> set[str]:
    return {"agent_id", "deficit", "stimulus_utility", "recent_arguments", "vsm_role"}


def test_enum_recognised() -> None:
    assert OrchestrationMode("langgraph_informed") is OrchestrationMode.LANGGRAPH_INFORMED
    print("[OK] OrchestrationMode.LANGGRAPH_INFORMED parsed from string.")


def test_router_select_with_mock_llm() -> None:
    router = LangGraphInformedRouter(
        base_url="http://invalid.local",
        api_key="dummy",
        model="dummy-model",
        seed=7,
    )

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="GOV-S2"))]

    with patch.object(router._client.chat.completions, "create", return_value=fake_resp) as m:
        chosen = router.select_next_agent_informed(
            discourse_context="Some prior arguments...",
            agent_features=[
                {"agent_id": "GOV-S1", "deficit": 0.5, "stimulus_utility": 0.1,
                 "recent_arguments": 0, "vsm_role": "S1"},
                {"agent_id": "GOV-S2", "deficit": 0.7, "stimulus_utility": 0.6,
                 "recent_arguments": 1, "vsm_role": "S2"},
            ],
        )
        assert chosen == "GOV-S2", f"expected GOV-S2 got {chosen!r}"
        prompt = m.call_args.kwargs["messages"][0]["content"]
        assert "stimulus_utility" in prompt
        assert "GOV-S2" in prompt
    print("[OK] LangGraphInformedRouter.select_next_agent_informed returns valid id.")


def test_router_fallback_on_invalid_response() -> None:
    router = LangGraphInformedRouter(
        base_url="http://invalid.local",
        api_key="dummy",
        model="dummy-model",
        seed=7,
    )
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="UNKNOWN"))]
    with patch.object(router._client.chat.completions, "create", return_value=fake_resp):
        chosen = router.select_next_agent_informed(
            discourse_context="",
            agent_features=[
                {"agent_id": "GOV-S1", "deficit": 0.5, "stimulus_utility": 0.0,
                 "recent_arguments": 0, "vsm_role": "S1"},
                {"agent_id": "OPP-S1", "deficit": 0.5, "stimulus_utility": 0.0,
                 "recent_arguments": 0, "vsm_role": "S1"},
            ],
        )
        assert chosen in {"GOV-S1", "OPP-S1"}
    print("[OK] Fallback to round-robin engages on unrecognised LLM output.")


def test_compute_router_features_schema() -> None:
    fake_llm = MagicMock()
    fake_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="GOV-S1"))]
    )
    with patch("langclaw.agent.OpenAI", return_value=fake_llm), \
         patch("langclaw.router.OpenAI", return_value=fake_llm):
        env = SotopiaEnvironment(
            base_url="http://invalid.local",
            api_key="dummy",
            model="dummy-model",
            max_iterations=1,
            api_hard_limit=1,
            initial_deficit=0.5,
            seed=42,
            orchestration_mode=OrchestrationMode.LANGGRAPH_INFORMED,
        )
        feats = env._compute_router_features()
        assert isinstance(feats, list) and len(feats) == len(env.agents)
        for f in feats:
            assert set(f.keys()) == _expected_features_schema(), f
            assert isinstance(f["deficit"], float)
            assert isinstance(f["recent_arguments"], int)
        print(f"[OK] _compute_router_features produced {len(feats)} entries with valid schema.")
        print(json.dumps(feats[:2], indent=2, ensure_ascii=False))


def main() -> int:
    test_enum_recognised()
    test_router_select_with_mock_llm()
    test_router_fallback_on_invalid_response()
    test_compute_router_features_schema()
    print("\nAll Phase 1.3 smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
