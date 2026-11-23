import torch
from attention_student import naive_strided_attention
from cuda.binding import strided_attention_forward

cases = [
    (1, 1, 1, 1, 1),    
    (2, 3, 17, 19, 3),  
    (2, 4, 8, 16, 32),  
    (2, 4, 1024, 64, 4),
    (1, 2, 128, 256, 1),
    (2, 5, 513, 129, 3),
]

for dtype, tol in [(torch.float64, 1e-6), (torch.float32, 1e-4), (torch.float16, 1e-2)]:
    for B, H, S, D, stride in cases:
        q = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
        k = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
        v = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
        ref = naive_strided_attention(q, k, v, stride)
        out = strided_attention_forward(q, k, v, stride)
        diff = (ref - out).abs().max().item()
        assert torch.allclose(ref, out, atol=tol, rtol=tol), f'FAIL {dtype} {(B,H,S,D,stride)} diff={diff}'
print('ALL EDGE CASES PASSED!')