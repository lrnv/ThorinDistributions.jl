# ThorinDistributions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/ThorinDistributions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lrnv.github.io/ThorinDistributions.jl/dev)
[![Build Status](https://github.com/lrnv/ThorinDistributions.jl/workflows/CI/badge.svg)](https://github.com/lrnv/ThorinDistributions.jl/actions)
[![Coverage](https://codecov.io/gh/lrnv/ThorinDistributions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lrnv/ThorinDistributions.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


This repo provides a julia package that implements several methods for dealing with gamma convolutions.

A gamma distribution is a distribution with pdf

$$f(x) = \frac{x^{α-1}e^{-x/θ}}{Γ(α)θ^α}$$

As Bondesson shows, based on Thorin work, the class of (weak limit of) independent convolutions of gamma distributions is quite large, closed with respect to independent addition and multiplication of random variables, and contains many interesting distributions.

We implement here a multivariate extensions of these results, and statistical estimation routines to allow for estimation of these distributions through a Laguerre expensions of their densities. 
