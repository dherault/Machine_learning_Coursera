# Week 11: Application Example: Photo OCR

## Photo OCR

### Problem Description and Pipeline

Optical Character Recognition

Photo OCR pipeline:
1 - Text detection
2 - Character segmentation
3 - Character classification

Idea: Break the problem into a sequence of different modules.

### Sliding Windows

To frame anything in a picture you need to know what width and height the frame will be.

We slide such a window from the top-left of the image, to the right, then the bottom, with a step size.
Then we increase (or decrease) the slidding windows size.

What if we don't know the width and height of the window ?

We do it with a square and get the black and white image from the video, apply to it an "expansion operator" to blur the results and get the correct width for each text zone

For each text zone, we perform a 1D sliding window to find splits between charaters.

### Getting Lots of Data and Artificial Data

Artificial data synthesis (Artificial charaters from fonts for ex.), Distortion.
Collect te data and label it yourself.
Crowd source (ex: Amazon Mechanical Turk)

Adding purely random meaningless noise does not usualy help.

1 - Make sure you have a low bias classifier before expanding the effort. (Plot learning curves)
Keep increasing the number of features/hidden units in neural network until you have a low bias classifier.

2 - "How much work would it be to get 10x as much data as we currently have?"

### Ceiling Analysis: What Part of the Pipeline to Work on Next

What part of the pipeline should we spend time to improve?

Idea: Compute the *overall* accuracy the the system (pipeline) given a test set.
Give the first module the correct outputs (given the inputs): you override it with a fake and purely accurate program.
Compute the overall accuracy again,
Then give the correct outputs to both the first and second module. Compute the overall accuracy.
