using ThorinDistributions
using Documenter

makedocs(;
    modules=[ThorinDistributions],
    authors="Oskar Laverny",
    repo="https://github.com/lrnv/ThorinDistributions.jl/blob/{commit}{path}#L{line}",
    sitename="ThorinDistributions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lrnv.github.io/ThorinDistributions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/ThorinDistributions.jl",
)
