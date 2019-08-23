# iceSheetsClimateIC2Tutorial
Brief tutorial on basic downloading and plotting of ICESat-2 data for the Ice Sheets and Climate group.
## WARNING: As of August 23rd, 2019 this repository is not stable. So please hold off on completing the preflight-checklist. 
## Preflight-Checklist
1. Create NASA EarthData login (https://urs.earthdata.nasa.gov/). This is required to retrieve ICESat-2 data.
2. If you have not already done this, install python3 with Anaconda (https://www.anaconda.com/distribution/).
3. Clone this repository. Navigate to the parent directory where you would like to have this repository live on your machine, then run `git clone https://github.com/EricKeenan/iceSheetsClimateIC2Tutorial.git` 
4. Navigate into the new repository, `cd iceSheetsClimateIC2Tutorial`. Then create  the required python environment, `conda env create -f ICESat-2.yml`. Then activate the python environment, `source activate ICESat-2.yml`. Python relies on a variety of open source tools. By activating this ICESat-2 environment, you are importing the python tools we need to download, process, and visualize ICESat-2 data. 
5. In order to ensure that all the python libraries are sucesfully imported, open `test_ICESat-2_environment.ipynb`. Then execute the code block with all the import statements. If no errors occur, then you have sucesfully imported all necessary python libraries. If you get error messages, you either need to troubelshoot, or seek advice from one of us! Good luck!   
