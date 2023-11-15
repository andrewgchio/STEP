# STEP: Semantics-Aware Sensor Placement for Monitoring Community-Scale Infrastructure

## Project Description<a id="description"></a>

*STEP* is a framework integrating *structural*, *behavioral*, and *semantic* aspects of an infrastructure to gain insight into computing a suitable sensor deployment. Together, these aspects provide *STEP* with the capability to partition a stormwater network, apply MILP optimization on the resulting pieces, and refine the solution using domain expert feedback. 

The *STEP* architecture consists of three main components:
* *Anomaly Generation*, which uses historical water quality grab sample data to construct sets of realistic anomalies. Our workflow makes anomaly profiles, and relates potential origins with nearby community-level semantics, which are leveraged in generating new potential anomalies for a system. 
* *Placement Optimization*, which fundamentally uses a divide-and-conquer approach towards proposing an ideal placement. The first step here looks to partition the stormwater network into smaller, more manageable pieces that can be optimized with MILP. The solutions are stitched together using simple heuristics. 
* *Placement Refinement*, which acknowledges that a proposed placement may initially not be ideal, and provides domain experts an opportunity to refine a placement and examine properties of the network. 

<p align="center">
    <img src="https://github.com/andrewgchio/STEP/assets/16398500/e6bc076e-3c69-4cad-92ae-271546d87ebe" alt="STEP Architecture">
</p>


## Requirements<a id="requirements"></a>

The following libraries should be all downloaded to use STEP: 

* [Python](https://www.python.org) (version >= 3.6.8)
* [numpy](https://numpy.org)(version >= 1.19.2)
* [pandas](https://pandas.pydata.org) (version >= 1.1.0)
* [networkx](https://networkx.org/) (version >= 3.2.1)
* [scikit-learn](https://scikit-learn.org/stable) (version >= 0.24.2)
* [tqdm](https://github.com/tqdm/tqdm) (version >= 4.62.3)
* [hymo](https://github.com/lucashtnguyen/hymo) (version >= 0.1.4)
* [flask](https://flask.palletsprojects.com/en/3.0.x/) (version >= 3.0)
* [shapely](https://pypi.org/project/shapely/)(version 2.0.2)
* [pyproj](https://pypi.org/project/pyproj/)(version 3.6.1)
* [psycop2](https://www.psycopg.org/docs/index.html)(version >= 2.9.9)

Note that most of these requirements can be fulfilled by using an [Anaconda](https://www.anaconda.com) environment. After installing Anaconda, open the Anaconda command line and execute: 
```
conda create --name step
conda activate step
conda install python numpy pandas networkx scikit-learn tqdm flask shapely pyproj psycop2
git clone https://github.com/lucashtnguyen/hymo
``` 
Note: hymo may need to be explicitly added to your path. 

A local Postgres database is used as the data source for STEP. Starting the PostgreSQL server service is all that is needed. 

## Using STEP

STEP is primarily set up to aid domain experts with the visualization of a network, 

## Running STEP

Start the PostgreSQL server as appropriate for your machine. 

Run the STEP dataserver: `python dashboard/dataserver.py`


## Citations: <a id="citations"></a>

If you use this project, please cite the following paper: 

Andrew Chio, Jian Peng, and Nalini Venkatasubramanian. 2023. STEP: Semantics-Aware Sensor Placement for Monitoring Community-Scale Infrastructure. In The 10th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys ’23), November 15–16, 2023, Istanbul, Turkey. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3600100.3623752

## Acknowledgements: <a id="acks"></a>

This work is supported by the UC National Laboratory Fees Research Program Grant No. L22GF4561, and National Science Foundation NSF Grants No. 1952247 and 2008993. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. 

## License: <a id="license"></a>

```
The MIT License (MIT)

Copyright (c) 2023 Andrew Chio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
