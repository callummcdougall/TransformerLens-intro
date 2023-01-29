import torch as t
from torchtyping import TensorType as TT
from typing import Callable
from transformer_lens import HookedTransformer, FactoredMatrix

def test_get_ablation_scores(ablation_scores: TT["layer", "head"], model: HookedTransformer, rep_tokens: TT["batch", "seq"]):

    import solutions
    
    ablation_scores_expected = solutions.get_ablation_scores(model, rep_tokens)

    t.testing.assert_close(ablation_scores, ablation_scores_expected)

    print("All tests in `test_get_ablation_scores` passed!")

def test_full_OV_circuit(OV_circuit: FactoredMatrix, model: HookedTransformer, layer: int, head: int):
    
        W_E = model.W_E
        W_OV = FactoredMatrix(model.W_V[layer, head], model.W_O[layer, head])
        W_U = model.W_U

        OV_circuit_expected = W_E @ W_OV @ W_U
    
        t.testing.assert_close(OV_circuit.get_corner(20), OV_circuit_expected.get_corner(20))
    
        print("All tests in `test_full_OV_circuit` passed!")

def test_pos_by_pos_pattern(pattern: TT["n_ctx", "n_ctx"], model: HookedTransformer, layer: int, head: int):

    import solutions

    W_pos = model.W_pos
    W_QK = model.W_Q[layer, head] @ model.W_K[layer, head].T
    
    score_expected = W_pos @ W_QK @ W_pos.T
    masked_scaled = solutions.mask_scores(score_expected / model.cfg.d_head ** 0.5)
    pattern_expected = t.softmax(masked_scaled, dim=-1)

    t.testing.assert_close(pattern[:50, :50], pattern_expected[:50, :50])

    print("All tests in `test_full_OV_circuit` passed!")

def test_decompose_attn_scores(decompose_attn_scores: Callable, q: t.Tensor, k: t.Tensor):

    import solutions

    decomposed_scores = decompose_attn_scores(q, k)
    decomposed_scores_expected = solutions.decompose_attn_scores(q, k)

    t.testing.assert_close(decomposed_scores, decomposed_scores_expected)

    print("All tests in `test_decompose_attn_scores` passed!")

def test_find_K_comp_full_circuit(find_K_comp_full_circuit: Callable, model: HookedTransformer):

    import solutions

    K_comp_full_circuit: FactoredMatrix = find_K_comp_full_circuit(model, 7, 4)
    K_comp_full_circuit_expected: FactoredMatrix = solutions.find_K_comp_full_circuit(model, 7, 4)

    assert isinstance(K_comp_full_circuit, FactoredMatrix), "Should return a FactoredMatrix object!"
    t.testing.assert_close(K_comp_full_circuit.get_corner(20), K_comp_full_circuit_expected.get_corner(20))

    print("All tests in `test_find_K_comp_full_circuit` passed!")

def test_get_comp_score(get_comp_score: Callable):

    import solutions

    W_A = t.rand(3, 4)
    W_B = t.rand(4, 5)

    comp_score = get_comp_score(W_A, W_B)
    comp_score_expected = solutions.get_comp_score(W_A, W_B)

    t.testing.assert_close(comp_score, comp_score_expected)