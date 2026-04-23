"""Smoke test for the custom vLLM agent loop with mocked generation."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.agent_loop.medserl_selfplay_agent import MedSerlSelfPlayAgentLoop


class FakeTokenOutput:
    def __init__(self, text: str, token_ids: list[int]):
        self.text = text
        self.token_ids = token_ids
        self.log_probs = [0.0] * len(token_ids)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return [ord(ch) for ch in text]
        return text

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(tok) for tok in token_ids)


class DictTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if tokenize:
            return {"input_ids": [ord(ch) for ch in text]}
        return text


class FakeServerManager:
    def __init__(self, outputs: list[FakeTokenOutput]):
        self.outputs = outputs
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        return self.outputs.pop(0)


class MedSerlSelfPlayAgentTest(unittest.TestCase):
    def _trainer_config(self):
        return SimpleNamespace(
            config={
                "actor_rollout_ref": {
                    "rollout": {
                        "response_length": 512,
                        "prompt_length": 256,
                    }
                }
            }
        )

    def test_tokenizer_dict_output_is_normalized(self):
        loop = MedSerlSelfPlayAgentLoop(
            trainer_config=self._trainer_config(),
            server_manager=FakeServerManager([]),
            tokenizer=DictTokenizer(),
            processor=None,
        )
        token_ids = loop._tokenize_messages(
            [{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
        )
        self.assertIsInstance(token_ids, list)
        self.assertTrue(token_ids)
        self.assertTrue(all(isinstance(tok, int) for tok in token_ids))

    def test_two_phase_game_emits_sparse_token_scores(self):
        async def run_test():
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                f.write(
                    '{"system_prompt":"You are an assessor.",'
                    '"user_template":"Review this note:\\n{sentences}"}'
                )
                detection_path = f.name

            trainer_config = self._trainer_config()
            tokenizer = FakeTokenizer()
            server_manager = FakeServerManager(
                outputs=[
                    FakeTokenOutput("2. The patient takes metformin.", [ord(c) for c in "2. The patient takes metformin."]),
                    FakeTokenOutput("CORRECT", [ord(c) for c in "CORRECT"]),
                ]
            )

            loop = MedSerlSelfPlayAgentLoop(
                trainer_config=trainer_config,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=None,
                detection_prompts_path=detection_path,
                judge_url="http://unused",
                judge_model="unused",
            )

            async def fake_judge_sentence_pair(**kwargs):
                del kwargs
                return {"verdict": "SAME", "judge_score": 1.0, "reason": "equivalent", "judge_output": "{}"}

            import scripts.self_play.agent_loop.medserl_selfplay_agent as module
            original_judge = module.judge_sentence_pair
            module.judge_sentence_pair = fake_judge_sentence_pair
            try:
                output = await loop.run(
                    {"temperature": 0.6, "top_p": 0.95},
                    prompt=[
                        {"role": "system", "content": "Inject."},
                        {"role": "user", "content": "1. Foo.\n2. The patient takes dimethylbiguanide."},
                    ],
                    extra_info={
                        "note_id": "note-1",
                        "mode": "benign",
                        "sentences": "1. Foo.\n2. The patient takes dimethylbiguanide.",
                        "error_type": "management",
                    },
                )
            finally:
                module.judge_sentence_pair = original_judge

            self.assertEqual(len(output.response_ids), len(output.response_mask))
            self.assertIn("generated_token_scores", output.extra_fields)
            scores = output.extra_fields["generated_token_scores"]
            self.assertEqual(len(scores), len(output.response_ids))
            self.assertGreater(output.extra_fields["injector_reward"], 0)
            self.assertGreater(output.extra_fields["assessor_reward"], 0)
            self.assertEqual(sum(1 for s in scores if s != 0), 2)

        asyncio.run(run_test())

    def test_parse_failure_still_runs_assessor_and_rewards_it(self):
        async def run_test():
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                f.write(
                    '{"system_prompt":"You are an assessor.",'
                    '"user_template":"Review this note:\\n{sentences}"}'
                )
                detection_path = f.name

            trainer_config = self._trainer_config()
            tokenizer = FakeTokenizer()
            server_manager = FakeServerManager(
                outputs=[
                    FakeTokenOutput("<think>reasoning only</think>", [ord(c) for c in "<think>reasoning only</think>"]),
                    FakeTokenOutput("CORRECT", [ord(c) for c in "CORRECT"]),
                ]
            )

            loop = MedSerlSelfPlayAgentLoop(
                trainer_config=trainer_config,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=None,
                detection_prompts_path=detection_path,
                judge_url="http://unused",
                judge_model="unused",
            )

            output = await loop.run(
                {"temperature": 0.6, "top_p": 0.95},
                prompt=[
                    {"role": "system", "content": "Inject."},
                    {"role": "user", "content": "1. Foo.\n2. The patient takes dimethylbiguanide."},
                ],
                extra_info={
                    "note_id": "note-2",
                    "mode": "benign",
                    "sentences": "1. Foo.\n2. The patient takes dimethylbiguanide.",
                    "error_type": "management",
                },
            )

            self.assertEqual(output.extra_fields["injector_outcome"], "parse_failure")
            self.assertEqual(output.extra_fields["judge_verdict"], "ABSTAIN")
            self.assertFalse(output.extra_fields["injector_format_valid"])
            self.assertEqual(output.extra_fields["assessor_ground_truth"], "CORRECT")
            self.assertEqual(output.extra_fields["assessor_output"], "CORRECT")
            self.assertGreater(output.extra_fields["assessor_reward"], 0)
            spans = output.extra_fields["turn_reward_spans"]
            self.assertEqual([span["role"] for span in spans], ["injector", "assessor"])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
