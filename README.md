# MSc thesis
 Thesis and presentation sources for MSc in Physics of Data @ UniPD by Marco Giunta

PDF Files:
* [MSc Thesis](../main/msc_thesis_Marco_Giunta "MSc thesis PDF")
* [MSc Presentation](../main/msc_presentation_Marco_Giunta "MSc presentation PDF")

Source codes on Github:
* [MSc Thesis](../main/thesis_source "MSc thesis source on github")
* [MSc Presentation](../main/presentation_source "MSc presentation source on github")

Up-to-date source codes/PDFs on overleaf:
* [MSc Thesis](https://www.overleaf.com/read/wpkbfsqdqcwp "MSc thesis source on overleaf")
* [MSc Presentation](https://www.overleaf.com/read/hqfcktymfvtr "MSc presentation source on overleaf")

## Title
Emulating Cosmological Likelihoods with Machine Learning

## Abstract
The process of Bayesian inference is a staple of modern science, as it allows us to learn from data in an intuitive and rigorous way. In the case of cosmology
Bayesian inference is often used to turn data about astrophysical objects into estimates of cosmological parameters whose importance is paramount to understanding the overall structure of the universe; 
such an inference relies on accurately evaluating complex likelihood functions, that are often expensive to
compute - requiring the use of specialized solvers. Despite the effort needed to
obtain individual function values many likelihoods of cosmological relevance
frequently turn out to be smooth, stable functions of their model parameters;
this has led to many attempts to exploit these remarkable mathematical properties in order to skip the expensive calculations. Several software frameworks
exist to allow one to approximate these functions, relying on a myriad of models, ranging from polynomial interpolation to Gaussian Processes to neural networks; yet these tools are often limited in scope, requiring the user to either
stick to provenly effective use-cases or retrain the model from scratch - thus
making the process of adapting likelihood emulation to new scenarios/datasets
needlessly time consuming. In this work we present COSMOLIME (Cosmological LIkelihood Machine learning Emulator), a model-agnostic, self-training, machine learning-based framework to emulate arbitrary likelihood functions in a
fully automated way. By providing COSMOLIME with a python function wrapping any exact likelihood solver the framework is fully autonomously able to
sample points as needed and optimize the hyperparameters of an arbitrary machine learning model. This allows the user to avoid the time consuming man-
ual process of tweaking models, checking the resulting performance, and deciding whether more simulated samples or different models altogether are needed.
With many powerful features such as the caching of intermediate results, pretty
logging, and options to parallelize both data generation and model optimization
CosmoLIME is a tool capable of substantially innovating the current cosmological landscape, as it can provide orders of magnitude speedups in both computer
and human time - thanks to the new DIY approach to emulation it offers.
