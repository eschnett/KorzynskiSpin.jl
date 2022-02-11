# Generate documentation with this command:
# (cd docs && julia --color=yes make.jl)

push!(LOAD_PATH, "..")

using Documenter
using KorzyńskiSpin

makedocs(; sitename="KorzyńskiSpin", format=Documenter.HTML(), modules=[KorzyńskiSpin])

deploydocs(; repo="github.com/eschnett/KorzyńskiSpin.jl.git", devbranch="main", push_preview=true)
