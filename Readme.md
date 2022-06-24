FTV code built on the basis of MyPTV

## How to install:

##### Requirements:

MyPTV requires you have Python 3 installed with pip, along with the Python packages: numpy, scipy, scikit-image, pandas, matplotlib, itertools

##### Installation:
###### Using `pip`

1) Open your terminal and change directory to the path of the code:
	`cd path/to/myptv` 
	
2) Finally, we use pip to install by using the following command: 
	`pip install .`
or 
	`pip install -r .\requirements.txt`

3) Optionally, parts of the code can be tested using pytest:
	`pytest ./tests/ -W ignore::RuntimeWarning`

4) Once this is done we are ready to go! You can now import MyPTV in your python code as usual. For example:
	`import myptv.imaging_mod`
or 	
   `from myptv import imaging_mod`

###### Using `conda` 

1) Install Anaconda or Miniconda and from the command shell inside the directory
where the package is downloaded:

	`conda env create -f environment.yml`
2) Activate the environment:

	`conda activate myptv`

3) Optionally, parts of the code can be tested using pytest:
	`pytest ./tests/ -W ignore::RuntimeWarning`

4) Once this is done we are ready to go! You can now import MyPTV in your python code as usual. For example:
	`import myptv.imaging_mod`
or 	
   `from myptv import imaging_mod`

## How to start?

Detailed instructions are given in the Manual, see `/user_manual/user_manual.pdf`.

## Who manages this project?

MyPTV was founded and is maintained by Ron Shnapp (ronshnapp@gmail.com). Contributions are most welcome. 

