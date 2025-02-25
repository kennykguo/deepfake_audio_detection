# XLS-R Model Overview

This README provides a summary of the XLS-R model and its potential usage in Deepfake Audio Detection Tasks.

## Summary

Acronym for eXtreme large Scale R-Transformer, XLS-R is a large-scale multilingual speech recognition model that aims to understand all human speech. [1]

It is Meta AI’s approach to developing a self-supervised learning of speech representation, where they utilized sources like parliamentary proceedings and audio books to train its ability to translate around ~ 128 languages.

It is based on Wav2Vec 2.0, an Automatic Speech Recognition model for self-supervised learning of speech representations. Unlike previous state-of-the-art models for multilingual language detection, Wav2Vec 2.0 utilizes speech instead of words. The model is accustomed to predict particular spoken words or phonemes.

## Capabilities and Relevance to Audio Processing Tasks

From the basis model wav2vec 2.0, XLS-R can be used for Multilingual ASR (Automatic Speech Recognition) tasks to transcribe speeches to text. Due to its training from half a million hours with different languages, it is additionally adept with translating these transcribed speeches between different languages. It also had use cases in detection speech-spoofing, which is when real, human speech audios are
artificially manipulated by third parties.

This usage for speech-spoofing gave rise to utilizing the model for Deepfake Audio Detection [3]. A group of students from Hubei Minzu University (China) proposed a method to fine tune the model with SLS (Sensitive Layer Selection) in order to tailor the XLS-R for this purpose [4]. The training consisted of first obtaining a pre-trained model of XLS-R, in conjunction with a Siamese Neural Network that performs
speech spoofing detection. Initial and intermediate layers of the pre-trained model are then extracted and put into a customer SLS classifier to determine whether the audio is real or deepfake.

## Input, X

The input X, is raw audio signals of unlabeled speech [3]. This input is processed by a convolutional encoder with several CNNs, with layer normalization and GELU activation function.

## Intermediate Output, Z

The process above creates Z, the latent (not completed) speech representations.

## Output, H

Z is then inputted into 24-layer transformer layers to produce the H, which are the contextualized representations of the detected, translated final output.

## Project Relevance

XLS-R can be directly utilized for deepfake audio detection.

In synthetic Deepfake audio generation like Text-To-Speech algorithms, the voice conversion process introduces vocoder artifacts. These artifacts are the source of unnatural and distorted voices, and cannot accurately mimic the real human speech flow and duration.

Part of XLS-R’s usage is to detect these vocoder artifacts from an inputted audio file. Unusual patterns like unnatural pauses, pitch glitches, and inconsistent inflections can be differentiated from the human speech that it was trained on. This is done through classifying contextualized representations (output H) of the different transformer layers by how much unnatural or distorted sound there are. Rather than the final output C of the model, Zhang's proposed
model utilizes the outputs of initial and intermediate (or hidden) layers. These ouputs are put into Zhang's Classification Model, which can be mathematically represented by the following expression.

𝐹𝜃(𝑥) = 𝐶𝜃𝑐(𝐻) = FC(maxpool(∑︁𝐿(𝑙=1) 𝛼ℎ))
𝛼 = Sigmoid(FC(avgpool(𝐻))) [3]

Where L represents the amount of transfermer layers, 𝛼 represents the layer weight, and ℎ represents the intermediate outputs.
Here, ℎ represent the intermediate outputs of XLS-R layers.

By training the XLS-R model through this proposed algorithm and samples of real and deepfake audio, it will become a powerful tool for deepfake audio detection.

## References

[1] "XLS-R: Self-supervised speech processing for 128 languages," Meta AI, Feb. 23, 2025. [Online]. Available: https://ai.meta.com/blog/xls-r-self-supervised-speech-processing-for-128-languages/. <br/>
[2] A. R. Babu et al., "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale," arXiv, Nov. 2021. [Online]. Available: https://arxiv.org/abs/2111.09296.<br/>
[3] Q. Zhang, S. Wen, and T. Hu, "Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier," OpenReview, Nov. 2021. [Online]. Available: https://openreview.net/pdf?id=acJMIXJg2u.
