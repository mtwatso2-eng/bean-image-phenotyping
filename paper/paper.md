---
title: 'Countess: a Shiny web app for automating nematode egg counts'
tags:
  - plant phenotyping
  - image analysis
  - nematode
  - python

authors:
 - name: Mark Watson
   orcid: 0009-0001-6607-880X
   affiliation: 1
 
affiliations:
 - name: University of California, Davis
   index: 1
   
date: 24 July 2025
bibliography: paper.bib
---
 
# Summary

Countess is a web app for automating counts of nematode eggs from counting slide images. Our hosted instance of the app can be accessed via a browser at [https://mtwatson-countess.share.connect.posit.cloud/](https://mtwatson-countess.share.connect.posit.cloud/).

# Statement of need

Nematode egg counts are an important measurement in nematode bioassays and monitoring for plant (Kalwa 2019), farm animal (Sréter 1994), and human (Hall, 1981) parasitology, but conventional manual counts are subject to low througput and repeatability (Akintayo et al., 2018). A variety of computer vision methods have been applied to nematode egg counting with moderate to high accuracy (Fraher et al., 2023; Holladay et al., 2016; Kalwa et al., 2019; Saikai 2024; Slusarewicz et al., 2016), but every implementation so far (including UI based implementations) require some amount of user code to run. This need for coding ability impedes access to improved nematode counts to non-coding nematologists. The Countess app is built to be accessible to researchers with no coding ability, as it requires no user code to run and has a simple web interface that only requires images from a previously described nematode egg imaging method (Fraher et al., 2024) as inputs. Also, the app uses cloud resources to run its image analysis pipeline, so powerful hardware on the user side is not required. For users who are interested in running the app locally, the app code can also be downloaded and run locally as-is.

# Design and usage

The Countess app is built on two main components: a Shiny (Chang et al., 2024) user interface (UI), and a backend image analysis algorithm. Both are implemented in Python. The Shiny user interface consists of a button to select a local directory containing nematode counting slide images, an output that displays uploaded images with egg countours drawn on them they are being counted, and a button to download a .csv file of counts. The UI requires minimal resources while remaining scalable to an arbitrary number of images, as it only loads one image to memory at a time and avoids app timeouts as long as counts are being performed. The protocol for collecting counting slide images using Chalex counting slides (Chalex, LLC, Park City, UT), along with the image analysis algorithm that the app uses to detect, classify, and count nematode eggs, are described in (Fraher et al., 2024). The imaging protocol is also summarized in the app. The web app takes a folder of counting slide images as input, runs an image analysis method on the input images, and returns a downloadable data frame of nematode counts corresponding to each image as output. The image analysis method is an updated version of the algorithm described in Fraher et al., 2024, which consists of the following steps: counting frame detection, egg-like object segmentation via thresholding implemented using OpenCV (Bradski 2000), egg classification via a convolutional neural network implemented in Pytorch (Paszke et al., 2019), and counting. The trained convolutional neural network is improved over the model described in Fraher et al., 2024 through the use of a larger training set of ~7000 segmentations vs the original ~3000 segmentations. We tested the web app on a dataset of ~1000 _Meloidogyne javanica_ slide images and obtained an egg count accuracy with R² = 0.984.

# Acknowledgements

This work was supported by the Specialty Crops Research Initiative Grant No. 2021-51181-35865 from the USDA National Institute of Food and Agriculture, the North Carolina Sweetpotato Commission Grant No. GRKN Proposal #22-05, and by a grant from Altria.

# References

Akintayo A, Tylka GL, Singh AK, Ganapathysubramanian B, Singh A, Sarkar S. A deep learning framework to discern and count microscopic nematode eggs. Scientific reports. 2018 Jun 14;8(1):9145.

Bradski G. The opencv library. Dr. Dobb's Journal: Software Tools for the Professional Programmer. 2000;25(11):120-3.

Chang, W., Cheng, J., Allaire, J., Sievert, C., Schloerke, B., Xie, Y., Allen, J., McPherson,
J., Dipert, A., & Borges, B. (2024). Shiny: Web Application Framework for R. https:
//doi.org/10.32614/CRAN.package.shiny

Fraher SP, Watson M, Nguyen H, Moore S, Lewis RS, Kudenov M, Yencho GC, Gorny AM. A Comparison of Three Automated Root-Knot Nematode Egg Counting Approaches Using Machine Learning, Image Analysis, and a Hybrid Model. Plant Disease. 2024 Sep 1;108(9):2625-9.

Hall A. Quantitative variability of nematode egg counts in faeces: a study among rural Kenyans. Transactions of the Royal Society of Tropical Medicine and Hygiene. 1981 Jan 1;75(5):682-7.

Holladay BH, Willett DS, Stelinski LL. High throughput nematode counting with automated image processing. BioControl. 2016 Apr;61(2):177-83.

Kalwa U, Legner C, Wlezien E, Tylka G, Pandey S. New methods of removing debris and high-throughput counting of cyst nematode eggs extracted from field soil. PLoS One. 2019 Oct 15;14(10):e0223386.

Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, Lin Z, Gimelshein N, Antiga L, Desmaison A. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems. 2019;32.

Posit team. (2025). Posit Connect: A Publishing Platform for R and Python. Posit Software, PBC. https://www.posit.co/.

Saikai KK, Bresilla T, Kool J, de Ruijter NC, Van Schaik C, Teklu MG. Counting nematodes made easy: leveraging AI-powered automation for enhanced efficiency and precision. Frontiers in Plant Science. 2024 Jun 26;15:1349209.

Slusarewicz P, Pagano S, Mills C, Popa G, Chow KM, Mendenhall M, Rodgers DW, Nielsen MK. Automated parasite faecal egg counting using fluorescence labelling, smartphone image capture and computational image analysis. International journal for parasitology. 2016 Jul 1;46(8):485-93.

Sréter T, Molnár V, Kassai T. The distribution of nematode egg counts and larval counts in grazing sheep and their implications for parasite control. International Journal for Parasitology. 1994 Feb 1;24(1):103-8.

