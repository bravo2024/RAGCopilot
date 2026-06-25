# Medical AI

## Medical image segmentation

Medical image segmentation partitions an image into regions of interest such as tumours or organs. U-Net is the standard architecture for biomedical segmentation.

## Digital pathology

Digital pathology scans whole-slide images of tissue at high resolution. AI models assist pathologists by detecting and grading abnormalities in H&E-stained slides.

## Gleason grading

Gleason grading is a histological scoring system for prostate cancer ranging from 3 to 5. AI models automate Gleason pattern recognition from biopsy slides.

## ISUP grade groups

ISUP grade groups condense Gleason scores into five prognostic categories. Grade Group 1 is Gleason 3+3 while Grade Group 5 is Gleason 9-10.

## Dice and IoU

Dice score and Intersection-over-Union are standard metrics for segmentation. Dice measures pixel overlap between predicted and ground-truth masks.

## Heatmap overlays

A heatmap overlay on a pathology slide shows where the model detects abnormality. High-intensity regions flag suspicious areas for pathologist review.

## Stain normalisation

Stain normalisation standardises colour variation in histology images caused by different labs and scanners. It improves model generalisation across institutions.

## Whole-slide image tiling

Whole-slide image tiling splits a gigapixel slide into smaller patches for processing. Tiling strategies use tissue-detection masks to skip background regions.

## Predictive toxicology

Predictive toxicology uses AI to forecast compound toxicity from chemical structure and multi-modal data, reducing reliance on animal testing.

## Medical AI safety

Medical AI safety disclaimers are essential: model outputs are research demonstrations and require clinician review before any clinical decision.
