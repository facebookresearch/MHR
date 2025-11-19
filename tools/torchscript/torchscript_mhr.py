import torch
from mhr.mhr import MHR

def run():
    # Create scripted version
    mhr_model = MHR.from_files(device=torch.device("cuda"), lod=1)
    batch_size = 2
    torch.manual_seed(0)
    shape_params = torch.randn(batch_size, 45).cuda() * 0.8
    model_params = 0.2 * (torch.rand(batch_size, 204 + 45 + 72) - 0.5).cuda()
    expr_params = torch.randn(batch_size, 72).cuda() * 0.3

    with torch.no_grad():
        trace = torch.jit.trace(
            mhr_model, (shape_params, model_params, expr_params), strict=True
        )

    torch.jit.save(trace, "./mhr_ts.pt")

if __name__ == "__main__":
    run()
