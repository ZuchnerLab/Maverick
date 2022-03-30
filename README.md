# Maverick
Maverick is a Mendelian approach to variant effect prediction built in keras. It leverages transformers to process a multi-modal set of inputs in order to predict whether a variant is benign, dominant pathogenic, or recessive pathogenic. 

This repository contains scripts to run inference with Maverick on VCF files aligned to GRCh37 in the "InferenceScripts" directory. This option would be best if you are setting up an installation of Maverick on a local workstation or cloud platform. For a less resource-intensive experience, try our [Maverick inference CoLab](https://colab.research.google.com/drive/1JyifjHLEWQesKzuCpDFZJoXNKJhg-4z8?usp=sharing). The CoLab will allow you to upload a VCF file and process it with Maverick right in your web browser for free. Currently, that notebook works well using a CPU or TPU backend, but may require some troubleshooting of the CUDA/cuDNN/Tensorflow versions in order to use the GPU backend.

This repository additionally contains python notebooks in the "Notebooks" directory demonstrating 1) how the training and testing sets used in the Maverick paper were generated, 2) how Maverick was trained, and 3) how to run score the variants in a VCF file with Maverick. Each of these notebooks are additionally available as Google CoLabs: [Generate Training and Test Sets](https://colab.research.google.com/drive/15FbOCsJ00j894PUBYdeCRDYpLMct8Wvv?usp=sharing), [Train Maverick](https://colab.research.google.com/drive/1bEjmt91epid9u_HqUfq5kor1uFg7OJ1z?usp=sharing), and [Maverick inference](https://colab.research.google.com/drive/1JyifjHLEWQesKzuCpDFZJoXNKJhg-4z8?usp=sharing).

The manuscript associated with Maverick is currently in submission for publication. This page will be updated with citation information when available. 

We have pre-computed Maverick scores for all possible autosomal missense and nonsense SNVs in the Gencode Basic V33 annotation of GRCh37. Several versions of those scores are avialable for download: 

[GRCh37](https://zuchnerlab.s3.amazonaws.com/VariantPathogenicity/MaverickResults_allSNVs_GRCh37.txt.gz)

[GRCh37 with scores from individual sub-models](https://zuchnerlab.s3.amazonaws.com/VariantPathogenicity/MaverickResults_allSNVs_GRCh37_withIndividualModelScores.txt.gz)

[Lifted over to GRCh38](https://zuchnerlab.s3.amazonaws.com/VariantPathogenicity/MaverickResults_allSNVs_GRCh38LiftOver.txt.gz)

[Lifted over to GRCh38 with scores from individual sub-models](https://zuchnerlab.s3.amazonaws.com/VariantPathogenicity/MaverickResults_allSNVs_GRCh38LiftOver_withIndividualModelScores.txt.gz)
