# Proposal Video Script

Lightweight Pothole Detection for Vehicle-Based Road Monitoring.
CSE36501 Deep Learning, Project Proposal, November 6, 2026.
Jung Young-chan (20221355), Jo Sumin (20221371).

Spoken script for the proposal video. The same text is kept in the `\note{}` blocks of
`slides/proposal.tex`; if you change one, change the other too.

---

## 1. Title

Hi everyone, we're Jung Young-chan and Jo Sumin. Our project is lightweight pothole detection
for cars. The question is: can a detector be accurate enough to be useful, but small and fast
enough to run on a car? Today we'll cover the problem and our datasets.

## 2. Motivation

So how are potholes found today? Road agencies inspect, or people report them. But nobody
really reports small potholes, and on rural roads it takes a while. Now imagine autonomous cars
are everywhere. Every car is already looking at the road, so it could spot a pothole and send
its location right away. The catch is, the car's computer is busy driving. A pothole detector is
just a side job, so it has to be light and fast.

## 3. Problem statement

Our task is to find potholes in road images and draw a box around each one, with just one
class. But for every model, we also report its size, its computation, and its speed, not just
accuracy. That gives us three questions. First, how small and fast can the detector get before
it stops being useful? Second, you can make a model cheaper with a smaller network, smaller
images, a lighter block, or run it at lower numerical precision, like FP16 or INT8. Which one hurts accuracy the least? And third,
does the efficient model still work in other countries, and in the rain?

## 4. Datasets

We use two datasets. RDD2022 is a big road damage dataset from six countries, about
forty-seven thousand images, all taken from vehicles, which is exactly our view. We only keep
the pothole class, D40, and we drop the China drone images, because they're shot from above.
MWPD, the Multi-Weather Pothole Detection dataset, is only for testing. It comes from different
sources and has real rain. But it doesn't say which images are rainy, so we labeled every image
ourselves.

## 5. What the data looks like

Here are some examples. On top is RDD2022. The camera is in the car and sees the whole street,
so potholes look small and far away. In the middle is MWPD in normal weather, mostly close-up
web photos where the pothole fills the frame. And at the bottom, in the rain, potholes are full
of water and reflect the sky, so they look pretty different.

## 6. Dataset analysis

Two things here. On the left, potholes are rare: only about one in ten RDD images has one, and
most of those are from Japan and India. On the right is the key point. About a quarter of RDD
potholes are tiny, under thirty-two pixels. Those are exactly the ones people don't report, and
the first ones a detector misses when we make it cheaper. So we report small-pothole accuracy
separately. One more thing: RDD frames next to each other look almost the same. So we group
every fifty frames into a block, and each block goes entirely into train, validation, or test.
That way, the test set never has near-copies of training images.

## 7. Evaluation design

Here's how we test. Set A is the clean RDD test split, our main accuracy number. B1 to B3 are
the same images with fake rain, from light to heavy. C and D are MWPD, in normal weather and in
real rain. Together, B, C and D answer our third question. And for every model, we look at two
things side by side: accuracy, including small potholes, and cost: size, computation, and speed
on a GPU and a CPU.

## 8. Plan & timeline

So where are we? The problem and the data are done. Next, we'll train baselines from tiny to
large, try the different ways to make them cheaper, and compare everything on accuracy-versus-cost
curves. We'll also measure speed on a CPU, since that's closer to what a car has. That's it from
us. Thanks for watching!
