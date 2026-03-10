# Dataset

## Crime Data in Ottawa

[Ottwa Police](https://data.ottawapolice.ca/search?layout=grid&sort=Title%7Ctitle%7Casc&tags=open%2520data)

## Street Lights in Ottawa

[Ottawa City](https://open.ottawa.ca/datasets/street-lights/about)

# Motivation

Numerous studies suggest that ambient lightings affect crime rate in urban environments.
Given the suggested studies, we further explore the correlations between street lights and crime rates.

Initially, we investigate the correlation coefficient by applying PCC analysis to justify the problem.
Then, since crime rates could vary by the ambient lightings, DiD is used to see the crime rates before and after the street light is installed.

## Spatial Transformer / Multi-Head Attention over Grid Patches

- Convert study area into a **raster grid** (e.g., 100m × 100m cells)
- Stack feature maps: one channel for light density, one for road density, one for land use, etc.
- Apply **multi-head self-attention over spatial patches** (similar to Vision Transformer / ViT)
- The attention heads can reveal which spatial regions and which feature channels (including lighting) the model focuses on when predicting crime in a target area
- More complex but increasingly used in urban computing research
