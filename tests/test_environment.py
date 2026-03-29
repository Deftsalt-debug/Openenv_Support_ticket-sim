#!/usr/bin/env python3
# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Test Suite

"""
Comprehensive tests for the Support Triage Environment.

Tests cover:
  - Model validation (Pydantic serialization/deserialization)
  - Ticket data generation (determinism, correct counts)
  - Grader scoring (exact match, partial credit, edge cases)
  - Environment lifecycle (reset → step → done → state)
  - Reward function properties (range, non-constant, partial progress)
  - API endpoint behavior (HTTP contract)
"""

import json
import os
import sys

import pytest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Category,
    Priority,
    ReviewDecision,
    Sentiment,
    Team,
    TriageAction,
    TriageObservation,
    TriageState,
    StepResult,
    TicketData,
)
from data.tickets import generate_ticket_pool, GroundTruth, LabeledTicket
from tasks.task_definitions import TASK_1, TASK_2, TASK_3, get_task, TASK_REGISTRY
from tasks.graders import (
    grade_action,
    compute_episode_score,
    score_priority,
    score_category,
    score_sentiment,
    score_team,
    score_decision,
    score_draft_response,
    score_escalation_reason,
)
from server.support_triage_environment import SupportTriageEnvironment
from client import SupportTriageClient


# ═════════════════════════════════════════════════════════════════════════════
# Model Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestModels:
    """Test Pydantic model validation and serialization."""

    def test_triage_action_minimal(self):
        """Action with only priority (Task 1 use case)."""
        action = TriageAction(priority=Priority.P1_HIGH)
        assert action.priority == Priority.P1_HIGH
        assert action.category is None
        d = action.model_dump(exclude_none=True)
        assert d == {"priority": "P1_HIGH"}

    def test_triage_action_full(self):
        """Action with all fields (Task 3 use case)."""
        action = TriageAction(
            priority=Priority.P0_CRITICAL,
            category=Category.SECURITY,
            sentiment=Sentiment.ANGRY,
            assigned_team=Team.SECURITY_TEAM,
            decision=ReviewDecision.ESCALATE,
            draft_response="We are investigating immediately.",
            escalation_reason="Active security breach.",
        )
        d = action.model_dump()
        assert d["priority"] == "P0_CRITICAL"
        assert d["decision"] == "ESCALATE"

    def test_triage_action_from_json(self):
        """Deserialize action from JSON (simulates wire format)."""
        raw = {"priority": "P2_MEDIUM", "category": "BILLING"}
        action = TriageAction(**raw)
        assert action.priority == Priority.P2_MEDIUM
        assert action.category == Category.BILLING

    def test_observation_serialization(self):
        """Observation round-trips through JSON."""
        obs = TriageObservation(
            done=False,
            reward=0.75,
            ticket=TicketData(
                ticket_id="TKT-001",
                subject="Test",
                body="Test body",
                customer_name="Alice",
                customer_tier="pro",
                timestamp="2026-03-15T10:00:00Z",
                channel="email",
            ),
            task_id="task_1",
            task_instructions="Test instructions",
            tickets_remaining=4,
        )
        d = obs.model_dump()
        obs2 = TriageObservation(**d)
        assert obs2.reward == 0.75
        assert obs2.ticket.ticket_id == "TKT-001"

    def test_state_serialization(self):
        """State round-trips through JSON."""
        state = TriageState(
            episode_id="ep-001",
            task_id="task_2",
            step_count=3,
            total_tickets=8,
            tickets_completed=3,
            cumulative_reward=2.1,
            max_possible_reward=8.0,
        )
        d = state.model_dump()
        state2 = TriageState(**d)
        assert state2.step_count == 3
        assert state2.cumulative_reward == 2.1


# ═════════════════════════════════════════════════════════════════════════════
# Data Generation Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDataGeneration:
    """Test synthetic ticket generation."""

    def test_deterministic_generation(self):
        """Same seed produces identical ticket pools."""
        pool1 = generate_ticket_pool("task_1", seed=42)
        pool2 = generate_ticket_pool("task_1", seed=42)
        assert len(pool1) == len(pool2)
        for t1, t2 in zip(pool1, pool2):
            assert t1.ticket.ticket_id == t2.ticket.ticket_id
            assert t1.ticket.subject == t2.ticket.subject

    def test_different_seeds_differ(self):
        """Different seeds produce different shuffle orderings."""
        pool1 = generate_ticket_pool("task_1", seed=42)
        pool2 = generate_ticket_pool("task_1", seed=99)
        # Same number of tickets
        assert len(pool1) == len(pool2)
        # Different ticket IDs (hash includes seed)
        ids1 = [t.ticket.ticket_id for t in pool1]
        ids2 = [t.ticket.ticket_id for t in pool2]
        assert ids1 != ids2  # Different seeds → different IDs

    def test_task_ticket_counts(self):
        """Each task generates the correct number of tickets."""
        assert len(generate_ticket_pool("task_1")) == 5
        assert len(generate_ticket_pool("task_2")) == 8
        assert len(generate_ticket_pool("task_3")) == 10

    def test_ground_truth_present(self):
        """Every ticket has valid ground-truth labels."""
        for task_id in ["task_1", "task_2", "task_3"]:
            pool = generate_ticket_pool(task_id)
            for lt in pool:
                gt = lt.ground_truth
                assert isinstance(gt.priority, Priority)
                assert isinstance(gt.category, Category)
                assert isinstance(gt.sentiment, Sentiment)
                assert isinstance(gt.assigned_team, Team)
                assert isinstance(gt.decision, ReviewDecision)

    def test_ticket_data_fields(self):
        """Ticket data has all required fields populated."""
        pool = generate_ticket_pool("task_1")
        for lt in pool:
            t = lt.ticket
            assert t.ticket_id
            assert t.subject
            assert t.body
            assert t.customer_name
            assert t.customer_tier in ("free", "pro", "enterprise")
            assert t.channel in ("email", "chat", "phone", "web_form")


# ═════════════════════════════════════════════════════════════════════════════
# Grader Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGraders:
    """Test scoring functions for correctness and properties."""

    # Priority scoring
    def test_priority_exact_match(self):
        assert score_priority(Priority.P0_CRITICAL, Priority.P0_CRITICAL) == 1.0

    def test_priority_off_by_one(self):
        assert score_priority(Priority.P1_HIGH, Priority.P0_CRITICAL) == 0.5

    def test_priority_off_by_two(self):
        assert score_priority(Priority.P2_MEDIUM, Priority.P0_CRITICAL) == 0.2

    def test_priority_off_by_three(self):
        assert score_priority(Priority.P3_LOW, Priority.P0_CRITICAL) == 0.0

    def test_priority_none(self):
        assert score_priority(None, Priority.P1_HIGH) == 0.0

    # Category scoring
    def test_category_exact_match(self):
        assert score_category(Category.BILLING, Category.BILLING) == 1.0

    def test_category_related(self):
        """TECHNICAL and BUG_REPORT are related → partial credit."""
        assert score_category(Category.TECHNICAL, Category.BUG_REPORT) == 0.4

    def test_category_unrelated(self):
        assert score_category(Category.BILLING, Category.SECURITY) == 0.0

    def test_category_none(self):
        assert score_category(None, Category.BILLING) == 0.0

    # Sentiment scoring
    def test_sentiment_exact(self):
        assert score_sentiment(Sentiment.ANGRY, Sentiment.ANGRY) == 1.0

    def test_sentiment_adjacent(self):
        assert score_sentiment(Sentiment.FRUSTRATED, Sentiment.ANGRY) == 0.5

    def test_sentiment_distant(self):
        assert score_sentiment(Sentiment.SATISFIED, Sentiment.ANGRY) == 0.0

    # Team scoring
    def test_team_exact(self):
        assert score_team(Team.ENGINEERING, Team.ENGINEERING) == 1.0

    def test_team_related(self):
        assert score_team(Team.ENGINEERING, Team.SECURITY_TEAM) == 0.4

    def test_team_unrelated(self):
        assert score_team(Team.BILLING_OPS, Team.ENGINEERING) == 0.0

    # Decision scoring
    def test_decision_exact(self):
        assert score_decision(ReviewDecision.ESCALATE, ReviewDecision.ESCALATE) == 1.0

    def test_decision_wrong(self):
        assert score_decision(ReviewDecision.RESPOND, ReviewDecision.ESCALATE) == 0.0

    # Draft response scoring
    def test_response_empty(self):
        gt = GroundTruth(
            priority=Priority.P1_HIGH, category=Category.TECHNICAL,
            sentiment=Sentiment.ANGRY, assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE,
            required_response_elements=["acknowledge_urgency"],
        )
        assert score_draft_response(None, gt, "Alice") == 0.0
        assert score_draft_response("", gt, "Alice") == 0.0

    def test_response_good_quality(self):
        gt = GroundTruth(
            priority=Priority.P1_HIGH, category=Category.TECHNICAL,
            sentiment=Sentiment.ANGRY, assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE,
            required_response_elements=["acknowledge_urgency", "escalation_confirmation"],
        )
        response = (
            "Hi Alice, I understand the urgency of this issue. "
            "I'm escalating this to our senior engineering team immediately."
        )
        score = score_draft_response(response, gt, "Alice")
        assert score > 0.7  # Should be high quality

    # Escalation reason scoring
    def test_escalation_not_required(self):
        gt = GroundTruth(
            priority=Priority.P3_LOW, category=Category.GENERAL,
            sentiment=Sentiment.NEUTRAL, assigned_team=Team.TIER1_SUPPORT,
            decision=ReviewDecision.RESPOND, escalation_required=False,
        )
        assert score_escalation_reason(None, gt) == 1.0

    def test_escalation_required_missing(self):
        gt = GroundTruth(
            priority=Priority.P0_CRITICAL, category=Category.TECHNICAL,
            sentiment=Sentiment.ANGRY, assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE, escalation_required=True,
        )
        assert score_escalation_reason(None, gt) == 0.0

    def test_escalation_required_provided(self):
        gt = GroundTruth(
            priority=Priority.P0_CRITICAL, category=Category.TECHNICAL,
            sentiment=Sentiment.ANGRY, assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE, escalation_required=True,
        )
        score = score_escalation_reason(
            "Production is down affecting all customers with revenue loss", gt,
        )
        assert score == 1.0

    # Full grade_action tests
    def test_grade_action_perfect_task1(self):
        """Perfect action for Task 1 should score 1.0."""
        pool = generate_ticket_pool("task_1", seed=42)
        for lt in pool:
            action = TriageAction(priority=lt.ground_truth.priority)
            result = grade_action(action, lt.ground_truth, "task_1")
            assert result.total_score == 1.0

    def test_grade_action_empty_scores_zero(self):
        """Empty action should score 0.0."""
        gt = GroundTruth(
            priority=Priority.P1_HIGH, category=Category.TECHNICAL,
            sentiment=Sentiment.ANGRY, assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE,
        )
        action = TriageAction()
        result = grade_action(action, gt, "task_1")
        assert result.total_score == 0.0
        assert any("Empty action" in p for p in result.penalties)

    def test_graders_not_constant(self):
        """Graders produce different scores for different inputs."""
        gt = GroundTruth(
            priority=Priority.P0_CRITICAL, category=Category.SECURITY,
            sentiment=Sentiment.ANGRY, assigned_team=Team.SECURITY_TEAM,
            decision=ReviewDecision.ESCALATE,
        )
        good = TriageAction(priority=Priority.P0_CRITICAL)
        bad = TriageAction(priority=Priority.P3_LOW)
        r1 = grade_action(good, gt, "task_1")
        r2 = grade_action(bad, gt, "task_1")
        assert r1.total_score != r2.total_score
        assert r1.total_score > r2.total_score

    def test_episode_score_computation(self):
        assert compute_episode_score([1.0, 0.5, 0.75]) == 0.75
        assert compute_episode_score([]) == 0.0
        assert compute_episode_score([1.0]) == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Environment Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestEnvironment:
    """Test the core environment lifecycle."""

    def setup_method(self):
        self.env = SupportTriageEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_id="task_1", seed=42)
        assert isinstance(obs, TriageObservation)
        assert obs.done is False
        assert obs.ticket is not None
        assert obs.task_id == "task_1"
        assert obs.tickets_remaining == 4  # 5 total, 1 shown

    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError, match="done"):
            self.env.step(TriageAction(priority=Priority.P1_HIGH))

    def test_full_episode_lifecycle(self):
        """Run a complete episode and verify lifecycle."""
        obs = self.env.reset(task_id="task_1", seed=42)
        assert not obs.done
        step_count = 0

        while not obs.done:
            action = TriageAction(priority=Priority.P2_MEDIUM)
            obs = self.env.step(action)
            step_count += 1
            assert obs.reward is not None
            assert 0.0 <= obs.reward <= 1.0

        assert obs.done
        assert step_count == 5  # Task 1 has 5 tickets

    def test_reward_range(self):
        """All rewards are in [0.0, 1.0]."""
        for task_id in ["task_1", "task_2", "task_3"]:
            self.env.reset(task_id=task_id, seed=42)
            obs = TriageObservation(done=False, reward=None)
            obs = self.env.reset(task_id=task_id, seed=42)
            while not obs.done:
                action = TriageAction(priority=Priority.P1_HIGH)
                obs = self.env.step(action)
                assert 0.0 <= obs.reward <= 1.0

    def test_state_tracking(self):
        """State accurately reflects episode progress."""
        self.env.reset(task_id="task_1", seed=42)

        state = self.env.state
        assert state.step_count == 0
        assert state.tickets_completed == 0
        assert not state.done

        self.env.step(TriageAction(priority=Priority.P1_HIGH))
        state = self.env.state
        assert state.step_count == 1
        assert state.tickets_completed == 1

    def test_cumulative_reward_increases(self):
        """Cumulative reward monotonically increases."""
        self.env.reset(task_id="task_1", seed=42)
        prev = 0.0
        obs = self.env.reset(task_id="task_1", seed=42)
        while not obs.done:
            action = TriageAction(priority=Priority.P2_MEDIUM)
            obs = self.env.step(action)
            assert obs.cumulative_reward >= prev
            prev = obs.cumulative_reward

    def test_all_tasks_completeable(self):
        """Each task can be completed successfully."""
        for task_id in ["task_1", "task_2", "task_3"]:
            obs = self.env.reset(task_id=task_id, seed=42)
            while not obs.done:
                action = TriageAction(
                    priority=Priority.P2_MEDIUM,
                    category=Category.TECHNICAL,
                    sentiment=Sentiment.NEUTRAL,
                    assigned_team=Team.ENGINEERING,
                    decision=ReviewDecision.RESPOND,
                    draft_response="Thank you for reaching out. We're looking into this.",
                )
                obs = self.env.step(action)
            assert obs.done

    def test_perfect_score_task1(self):
        """Perfect actions yield a perfect score."""
        pool = generate_ticket_pool("task_1", seed=42)
        obs = self.env.reset(task_id="task_1", seed=42)

        # We need to match the shuffled order
        # Instead, just check that a correct priority gets 1.0
        # by using the ticket from the observation
        rewards = []
        for i, lt in enumerate(pool):
            action = TriageAction(priority=lt.ground_truth.priority)
            obs = self.env.step(action)
            rewards.append(obs.reward)

        assert all(r == 1.0 for r in rewards)

    def test_step_after_done_raises(self):
        """Stepping after episode end raises RuntimeError."""
        obs = self.env.reset(task_id="task_1", seed=42)
        while not obs.done:
            obs = self.env.step(TriageAction(priority=Priority.P2_MEDIUM))

        with pytest.raises(RuntimeError, match="done"):
            self.env.step(TriageAction(priority=Priority.P2_MEDIUM))

    def test_multiple_resets(self):
        """Environment can be reset multiple times."""
        for _ in range(3):
            obs = self.env.reset(task_id="task_1", seed=42)
            assert not obs.done
            assert obs.ticket is not None

    def test_different_tasks_different_ticket_counts(self):
        """Different tasks have different numbers of tickets."""
        obs1 = self.env.reset(task_id="task_1")
        remaining1 = obs1.tickets_remaining + 1  # +1 for current

        obs2 = self.env.reset(task_id="task_2")
        remaining2 = obs2.tickets_remaining + 1

        obs3 = self.env.reset(task_id="task_3")
        remaining3 = obs3.tickets_remaining + 1

        assert remaining1 < remaining2 < remaining3

    def test_feedback_provided_each_step(self):
        """Feedback is provided after each step (not just at end)."""
        obs = self.env.reset(task_id="task_1", seed=42)
        obs = self.env.step(TriageAction(priority=Priority.P1_HIGH))
        assert obs.step_feedback is not None
        assert len(obs.step_feedback) > 0


# ═════════════════════════════════════════════════════════════════════════════
# Client Tests (local mode)
# ═════════════════════════════════════════════════════════════════════════════

class TestClient:
    """Test the client in local mode."""

    def test_local_client_lifecycle(self):
        with SupportTriageClient.local() as client:
            obs = client.reset(task_id="task_1", seed=42)
            assert not obs.done
            assert obs.ticket is not None

            obs = client.step(TriageAction(priority=Priority.P2_MEDIUM))
            assert obs.reward is not None

            state = client.state()
            assert state.step_count == 1

    def test_local_client_full_episode(self):
        with SupportTriageClient.local() as client:
            obs = client.reset(task_id="task_1", seed=42)
            while not obs.done:
                obs = client.step(TriageAction(priority=Priority.P1_HIGH))
            assert obs.done


# ═════════════════════════════════════════════════════════════════════════════
# Task Definition Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTaskDefinitions:
    """Test task definitions are complete and consistent."""

    def test_all_tasks_registered(self):
        assert "task_1" in TASK_REGISTRY
        assert "task_2" in TASK_REGISTRY
        assert "task_3" in TASK_REGISTRY

    def test_task_lookup(self):
        t = get_task("task_1")
        assert t.task_id == "task_1"
        assert t.title == "Priority Classification"

    def test_invalid_task_raises(self):
        with pytest.raises(KeyError):
            get_task("task_99")

    def test_difficulty_progression(self):
        from models import TaskDifficulty
        assert TASK_1.difficulty == TaskDifficulty.EASY
        assert TASK_2.difficulty == TaskDifficulty.MEDIUM
        assert TASK_3.difficulty == TaskDifficulty.HARD

    def test_ticket_count_progression(self):
        assert TASK_1.num_tickets < TASK_2.num_tickets < TASK_3.num_tickets

    def test_evaluated_fields_progression(self):
        """Harder tasks evaluate more fields."""
        assert len(TASK_1.evaluated_fields) < len(TASK_2.evaluated_fields)
        assert len(TASK_2.evaluated_fields) < len(TASK_3.evaluated_fields)

    def test_instructions_non_empty(self):
        for task in [TASK_1, TASK_2, TASK_3]:
            assert len(task.instructions) > 100


# ═════════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
