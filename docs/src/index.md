```@meta
CurrentModule = ThorinDistributions
```

# ThorinDistributions

A gamma distribution is a distribution with pdf

$$f(x) = \frac{x^{α-1}e^{-x/θ}}{Γ(α)θ^α}$$

As Bondesson shows, based on Thorin work, the class of (weak limit of) independent convolutions of gamma distributions is quite large, closed with respect to independent addition and multiplication of random variables, and contains many interesting distributions.

We implement here a multivariate extensions of these results, and statistical estimation routines to allow for estimation of these distributions through a Laguerre expensions of their densities. 

```@index
```

```@autodocs
Modules = [ThorinDistributions]
```
