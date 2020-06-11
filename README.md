#### Author: Rahul Mehta
#### Alpha Version: Started June 1, 2020
#### Notes:
1. Phase 1 in progress — Tracking occupancy of a parking lot from a singular clean, sample image.
2. Analysis of Work: Improvements from baseline open-sources include  binary conversion of image before canny-edge detection, automated houghlines into x-clustering. For automatic x-clustering, average of houghlines x-coords (x1,x2) were  smoothed, plotted, and local maximas were taken to identify parking lanes.
3. Limitations of Phase 1: Region of interest was manually defined; otherwise, data became somewhat noisy for proper maxima analysis. Also, width of parking spots were also manually defined.
4. Two VGG16 models were used and compared, accuracy scores are in rccn.py
#### Acknowledgements:
1. http://cs231n.stanford.edu/reports/2016/pdfs/280_Report.pdf for providing an in-depth analysis of the topic.
2. Huge thanks to Priya Dwivedi, who worked on a similar problem and allowed some of her work (specifically, the R-CNN model) to be open-sourced and serve as a baseline for this project. 
3.  Also, thanks to Adam Geitgey. Article: https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400.
