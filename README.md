#### Author: Rahul Mehta
#### Alpha Version: Started June 1, 2020
#### Notes:
1. Phase 2 in progress — Scraping data from satellite imagery sources (NAIP) and gleaning occupancy metrics. Phase 1 completed — Tracking occupancy of a parking lot from a clean, sample image.
2. Analysis of Work: Improvements include binary conversion of image (and Gaussian smoothing) before Canny-Edge Detection, a more robust houghline probabilistic transformation to support differences in parking lots from aerial imagery, and automated houghline transformation into x-clustering to identify parking lanes. For automatic x-clustering, average of houghlines x-coords (x1,x2) were  smoothed, plotted, and local maximas were taken to identify parking lanes. Despite differences in lots, bounding box creation stayed the same as from that height, the model is adequate in determining the approximate occupancy percentage even if the lines are a few pixels off. For Phase 2, list of stores were scraped from websites' store locator (can be repurposed from different retailers). Cross-referenced with available county & state images for certain years and random sampling of stores taken from the resulting filtered list. 
3. Limitations of Phase 1: Region of interest was manually defined; otherwise, data became somewhat noisy for proper maxima analysis. Also, width of parking spots were manually defined (Reasoning in #2). Two VGG16 models were used and compared, accuracy scores are in rccn.py and one was chosen for use.
4. Limitations of Phase 2: NAIP database does not have multiple images an year. Leaned on Law of Large Numbers, implying that random sampling can provide an accurate approximation of occupancy for that year. 
#### Pipeline:
Scrape stores from store locator -> cross-reference with NAIP database -> take a random sample of stores for each year in series of years -> filter county images to coordinates of stores -> run occupancy model -> occupancy metric for each year.
##### Manual Changes to Pipeline for Repurposement:
1. Amend scrape_stores() in webscraper.py for different store locators.
#### Acknowledgements:
1. http://cs231n.stanford.edu/reports/2016/pdfs/280_Report.pdf for providing an in-depth analysis of the topic.
2. Huge thanks to Priya Dwivedi, who worked on a similar problem and allowed some of her work (specifically, the R-CNN model) to be open-sourced and serve as a baseline for this project. 
3.  Also, thanks to Adam Geitgey. Article: https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400.
