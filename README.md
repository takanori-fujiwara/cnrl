## cNRL: Contrastive Network Representation Learning

About
-----
* Python3 implemetation of cNRL from Fujiwara et al., 2022.
  * Fujiwara et al., Network Comparison with Interpretable Contrastive Network Representation Learning, Journal of Data Science, Statistics, and Visualisation, forthcoming (arXiv:2005.12419).
******

Requirements
-----
* Python3
* OS: macOS or Linux
  * Note: cNRL itself is available from Windows but DeepGL used in the above paper is not available from Windows.

* To run sample.py, also ccPCA and DeepGL.
  * ccPCA: https://github.com/takanori-fujiwara/ccpca
  * DeepGL: https://github.com/takanori-fujiwara/deepgl
* Note: Tested on macOS Catalina and Ubuntu 20.0.4 LTS.
******

Setup
-----
* Install with pip3. Move to the directory of this repository. Then,

    `pip3 install .`

******

Usage
-----
* Import an installed module from python (e.g., `from cnrl import CNRL`). See sample.py for examples.

* For detailed documentations, please see doc/index.html or directly see comments in cnrl.py.

******

How to Cite
-----
Please cite:
Fujiwara et al., Network Comparison with Interpretable Contrastive Network Representation Learning. Journal of Data Science, Statistics, and Visualisation, forthcoming (arXiv:2005.12419).

Also, there is one more closely related work:
Fujiwara et al., A Visual Analytics Framework for Contrastive Network Analysis. In Proc. IEEE Conference on Visual Analytics Science and Technology (VAST), 2020.
