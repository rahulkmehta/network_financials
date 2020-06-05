#### Author: Rahul Mehta
#### Version: Alpha; Started June 1, 2020
#### Notes:
> Phase 1 in progress — Tracking occupancy of a parking lot from a singular clean, sample image.
> Analysis of Work: Improvements from baseline open-sources include better binary conversion, canny-edge detection performed before houghlines, and automatic partitition within x-clusters (previously manual) among other things. For automatic clustering, average of houghlines (x1,x2) was smoothed, plotted, and local maximas were taken to simulate parking lanes
> Limitations of Phase 1: Region of Interest was manually defined; otherwise, data became too messy for proper maxima analysis.
#### Acknowledgements:
> http://cs231n.stanford.edu/reports/2016/pdfs/280_Report.pdf for providing an in-depth analysis of the topic.
> Huge thanks to Priya Dwivedi, who worked on a similar problem and allowed her work to be open-sourced and serve as a baseline for this project. 
> Also, thanks to Adam Geitgey. Article: https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400.