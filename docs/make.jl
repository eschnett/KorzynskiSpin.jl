# Generate documentation with this command:
# (cd docs && julia --project=. make.jl)

using Documenter
using KorzynskiSpin

makedocs(;
    sitename="KorzynskiSpin.jl",
    format=Documenter.HTML(),
    modules=[KorzynskiSpin],
    pages=["Home" => "index.md", "The algorithm" => "algorithm.md", "Related work" => "related_work.md"],
)

deploydocs(; repo="github.com/eschnett/KorzynskiSpin.jl.git", devbranch="main", push_preview=true)
