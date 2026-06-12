# Generate documentation with this command:
# (cd docs && julia --project=. make.jl)

using Documenter
using KorzyńskiSpin

makedocs(;
    sitename="KorzyńskiSpin.jl",
    format=Documenter.HTML(),
    modules=[KorzyńskiSpin],
    pages=["Home" => "index.md", "The algorithm" => "algorithm.md", "Related work" => "related_work.md"],
)

deploydocs(; repo="github.com/eschnett/KorzynskiSpin.jl.git", devbranch="main", push_preview=true)
