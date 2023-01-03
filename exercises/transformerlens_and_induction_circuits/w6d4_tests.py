import torch as t
import torch.nn as nn

def test_qk_attn(QK_attn, W_QK, attn_input, attn_pattern):
    with t.inference_mode():
        QK_attn_pattern = QK_attn(W_QK, attn_input)
        t.testing.assert_close(QK_attn_pattern, attn_pattern, atol=1e-3, rtol=0)


def test_ov_result_mix_before(OV_result_mix_before, W_OV, residual_stream_pre, attn_pattern, expected_head_out):
    with t.inference_mode():
        actual_head_out = OV_result_mix_before(W_OV, residual_stream_pre, attn_pattern)
        t.testing.assert_close(actual_head_out, expected_head_out, atol=1e-1, rtol=0)


def test_ov_result_mix_after(OV_result_mix_after, W_OV, residual_stream_pre, attn_pattern, expected_head_out):
    with t.inference_mode():
        actual_head_out = OV_result_mix_after(W_OV, residual_stream_pre, attn_pattern)
        t.testing.assert_close(actual_head_out, expected_head_out, atol=1e-1, rtol=0)


def test_logit_attribution(logit_attribution, model, cache, tokens, logits):
    with t.inference_mode():
        batch_index = 0
        embed = cache["hook_embed"]
        l1_results = cache["blocks.0.attn.hook_result"]
        l2_results = cache["blocks.1.attn.hook_result"]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens)
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[batch_index, t.arange(len(tokens[0]) - 1), tokens[batch_index, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-2, rtol=0)