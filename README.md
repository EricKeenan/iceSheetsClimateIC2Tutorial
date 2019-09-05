# iceSheetsClimateIC2Tutorial
Brief tutorial on basic downloading and plotting of ICESat-2 data for the Ice Sheets and Climate group.
## Preflight-Checklist
1. Create NASA EarthData login (https://urs.earthdata.nasa.gov/). This is required to retrieve ICESat-2 data.

2. If you have not already done this, install python3 with Anaconda (https://www.anaconda.com/distribution/).

3. Clone this repository. Navigate to the parent directory where you would like to have this repository live on your machine, then run `git clone https://github.com/EricKeenan/iceSheetsClimateIC2Tutorial.git` 

4. Navigate into the new repository, `cd iceSheetsClimateIC2Tutorial`. Then create  the required python environment, `conda env create -f ICESat-2.yml` (this will take a few minutes). Next, activate the python environment, `source activate remote2`. Finally, add the environment to the list of available kernels. `python -m ipykernel install --user --name remote2 --display-name remote2`

The beauty of Python is that it is community developed and open source, thus if something isn't working, it is possible to fix it yourself (or ask an expert to do it for you)! However, this also means that different users very likely have different combinations of packages and versions, thus complicating the sharing of code (sigh...). In order to get around this we will use conda environments. By using our conda enviornment (ICESat-2.yml) we synchorize our version of Python! 

For non-macOS users: I had issues using this environment with linux. If you plan to use linux or windows, you may encounter issues with `conda env create -f ICESat-2.yml`. If so, check this out (https://johannesgiorgis.com/sharing-conda-environments-across-different-operating-systems/) or come to one of us. 

5. In order to ensure that all the python libraries are sucesfully imported, open `notebooks/test_ICESat-2_environment.ipynb` and activate the remote2 kernel. Then execute the code block with all the import statements. If no errors occur, then you have sucesfully imported all necessary python libraries. If you get error messages, you either need to troubelshoot, or seek advice from one of us! Good luck!  

## Tutorial Creation To Do List
1. Compile ICESat-2 data product overview materials. 
2. Test on external machine.  
