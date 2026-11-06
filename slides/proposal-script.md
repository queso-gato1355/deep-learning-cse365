# Proposal Video Script

Lightweight Pothole Detection for Vehicle-Based Road Monitoring.
CSE36501 Deep Learning, Project Proposal, November 6, 2026.
Jung Young-chan (20221355), Jo Sumin (20221371).

Extracted from the `\note{}` blocks in `slides/proposal.tex`. Edit the notes there and re-extract
so the slides and the script stay in sync.

---

## 1. Title

Hello, we are Jung Young-chan and Jo Sumin. Our project is lightweight pothole detection for
vehicle-based road monitoring: can a detector be accurate enough, yet small and fast enough to
run on a car? Today we cover the problem and the datasets.

## 2. Motivation

Today, potholes are found by road-agency inspections or by citizen reports. But small potholes
are rarely reported, and on rural roads reports come late. Once autonomous cars are common,
every car is a road sensor: it could detect a pothole and send its location to the road agency
right away. The catch is that the car's computer is busy driving, so a pothole detector, as a
side task, must be efficient and fast.

## 3. Problem statement

Formally, the task is single-class pothole detection with bounding boxes under a compute
constraint, so every result is reported as accuracy together with model size, GFLOPs and speed.
This gives three research questions. First, how small and fast can the detector get before
accuracy becomes useless? Second, a model can be made cheaper with a smaller network, smaller
input images, a lighter module, or lower numerical precision. Which one hurts accuracy least?
Third, does the efficient model still work in other countries and in the rain?

## 4. Datasets

We use two datasets. RDD2022 is a large multi-country road damage dataset, about forty-seven
thousand images captured from vehicles, which is exactly our viewpoint. We keep only the pothole
class, D40, and drop the China drone subset because it is shot from above. MWPD, the
Multi-Weather Pothole Detection dataset, is used only for testing: it comes from a different
source and includes real rain. It has no weather labels, so we tagged every image ourselves.

## 5. What the data looks like

Here is what the data looks like. In RDD2022, on top, the camera sits in a car and sees the
whole street, so potholes are small and far away. MWPD in normal weather, in the middle, is
mostly close-up web photos where the pothole fills the frame. In the rain, at the bottom,
potholes are filled with water and reflect the sky, so they look quite different.

## 6. Dataset analysis

Two observations. On the left, potholes are rare: only about one in ten annotated RDD images
contains one, mostly from Japan and India. On the right, the key point: about a quarter of RDD
potholes are small objects, under thirty-two pixels at the usual input size. Those are the
potholes people never report, and the first a detector loses when we make it cheaper, so we
report small-object accuracy separately. Finally, RDD has consecutive frames, so we split in
blocks of fifty frames to keep near-identical images out of both train and test.

## 7. Evaluation design

Set A, our clean RDD test split, measures the main accuracy. B1 to B3 are the same images with
synthetic rain of increasing strength, and C and D are MWPD in normal weather and in real rain;
together they answer the third question. Every model is reported on two axes: accuracy,
including small objects, and cost, meaning parameters, GFLOPs and speed on a GPU and a CPU.

## 8. Plan & timeline

The problem and the data are done. Next we train baselines from tiny to large, apply the
efficiency levers, and compare them with accuracy-versus-cost curves, including speed on a CPU
as a stand-in for in-car hardware. Thank you.
