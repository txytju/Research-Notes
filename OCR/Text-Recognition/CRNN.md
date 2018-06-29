## Keywords

- image-based sequence recognition 

## Meta information

- Year 2015
- Conference 

## Contribution 

A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified frame- work, is proposed. 

## Advantages

Compared with previous systems for scene text recognition, the proposed architecture possesses four distinctive properties:

- It is end-to-end trainable, in contrast to most of the existing algorithms whose compo- nents are separately trained and tuned.
- It naturally handles sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization.
- It is not confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks.
- It generates an effective yet much smaller model, which is more practical for real-world ap- plication scenarios.

## Background

In summary, current(before this paper) systems based on DCNN can not be directly used for image-based sequence recognition. 

- Why Deep Convolutional Neural Networks (DCNN) can't be applied to text recognition
  - 1
  - 2
- Some methods that are applied in text recognition task
  - 1
  - 2

## Experiments

The experiments on standard bench-marks, including the IIIT-5K, Street View Text and ICDAR datasets, demonstrate the superiority of the proposed algo- rithm over the prior arts. 