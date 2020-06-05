#### Author: Rahul Mehta
#### Version: Alpha; Started June 1, 2020
#### Notes:
> Phase 1 in progress — Tracking occupancy of a parking lot from a clean, sample image.
> Analysis: Improvements from baseline open-sources include better binary conversion, canny-edge detection performed before houghlines, and automatic partitition within x-clusters (previously manual). For automatic clustering, mean-shift was used.
#### Acknowledgements:
> http://cs231n.stanford.edu/reports/2016/pdfs/280_Report.pdf for providing an in-depth analysis of the topic.
> Huge thanks to Priya Dwivedi, who worked on a similar problem and allowed her work to be open-sourced and serve as a baseline for this project. 
> Also, thanks to Adam Geitgey. Article: https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400.