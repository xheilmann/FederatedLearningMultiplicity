import wandb

api = wandb.Api()

id = [
    "75yd4f0o",
    "uds9pq4e",
    "8xws5bxt",
    "50i7g972",
    "846otkrv",
    "4q3wmkr3",
    "iwokaabn",
    "3mlrwpkw",
    "i1ehvcp5",
    "n6goarl6",
    "8plehq3r",
    "3zpuym23",
    "6a9jy9x0",
    "p80banao",
    "xr364lgy",
    "u59a5w15",
    "god2l9fd",
    "lunsipmu",
    "icq4e6vs",
    "03x8ouz5",
    "eu8ymp9o",
    "glpul6yp",
    "b5if24gq",
    "zj932tkf",
    "ugf6pt4a",
    "5z10px2f",
    "3h7x72xz",
    "8vxrmwre",
]

for i in id:
    run = api.run(f"lucacorbucci/DutchMultiplicityEvaluation_50_clients/{i}")
    run.file("wandb-summary.json").download(replace=True, root=f"dutch_results/{run.name}")

api_list = ["lucacorbucci/IncomeMultiplicityEvaluation"]
for a in range(len(api_list)):
    # Project is specified by <entity/project-name>
    runs = api.runs(api_list[a])

    runs_df = {}
    rc_runs_df = {}

    for run in runs:
        # print(run.created_at)
        if run.created_at >= "2025-11-28T09:54:16.669007Z":
            run.file("wandb-summary.json").download(replace=True, root=f"income_results/{run.name}")
