"""Tests for SDK model types."""

import pytest

from bandito.models import Arm, PullResult


class TestArm:
    def test_convenience_properties(self):
        arm = Arm(
            arm_id=1,
            model_name="gpt-4",
            model_provider="OpenAI",
            system_prompt="Be helpful",
            is_prompt_templated=False,
        )
        assert arm.model == "gpt-4"
        assert arm.prompt == "Be helpful"

    def test_frozen(self):
        arm = Arm(arm_id=1, model_name="gpt-4", model_provider="OpenAI",
                  system_prompt="test", is_prompt_templated=False)
        with pytest.raises(AttributeError):
            arm.model_name = "gpt-5"


class TestPullResult:
    def test_reach_through_properties(self):
        arm = Arm(arm_id=1, model_name="gpt-4", model_provider="OpenAI",
                  system_prompt="Be helpful", is_prompt_templated=False)
        result = PullResult(
            arm=arm,
            event_id="abc-123",
            bandit_id=1,
            bandit_name="test",
            scores={1: 0.5},
        )
        assert result.model == "gpt-4"
        assert result.prompt == "Be helpful"
        assert result.event_id == "abc-123"

    def test_frozen(self):
        arm = Arm(arm_id=1, model_name="gpt-4", model_provider="OpenAI",
                  system_prompt="test", is_prompt_templated=False)
        result = PullResult(arm=arm, event_id="x", bandit_id=1,
                            bandit_name="test", scores={})
        with pytest.raises(AttributeError):
            result.bandit_id = 2
