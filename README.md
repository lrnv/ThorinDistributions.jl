# ThorinDistributions

[![DOI](https://zenodo.org/badge/315324274.svg)](https://zenodo.org/badge/latestdoi/315324274)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/ThorinDistributions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lrnv.github.io/ThorinDistributions.jl/dev)
[![Build Status](https://github.com/lrnv/ThorinDistributions.jl/workflows/CI/badge.svg)](https://github.com/lrnv/ThorinDistributions.jl/actions)
[![Coverage](https://codecov.io/gh/lrnv/ThorinDistributions.jl/branch/main/graph/badge.svg?token=WoTkyO2rWU)](https://codecov.io/gh/lrnv/ThorinDistributions.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


This Julia package implements some tools to work with and around generalized gamma convolutions and their multivariate extensions. 

A non-exhaustive list of included features:
- Density evaluation through Moschopoulos, Mathai, and novel Laguerre series in univariate case
- Density evaluation in the multivariate case through Laguerre series
- Laverny's Estimation through an L2 loss on density in the Laguerre basis
- Estimation through a shifted-cumulant-based loss 
- Random number generation and Distributions overloading infrastructure
- Miles, Furman and Kuznetsov projection algorithm


This is still a WIP and the documentation is sparse, as are the test coverage. Use with caution.
