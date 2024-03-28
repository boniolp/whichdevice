<h1 align="center">DeviceScope</h1>

<p align="center">
<img width="350" src="./Figures/DeviceScope_demo.gif"/>
</p>

<h2 align="center">An interactive tool to browse, detect and localize appliance patterns in electrical consumption time series</h2>

<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/boniolp/whichdevice"> <img alt="GitHub issues" src="https://img.shields.io/github/issues/boniolp/whichdevice">
</p>
</div>

<p align="center"><a href="https://whichdevice.streamlit.app/">Try our demo</a></p>

DeviceScope is a Python-based web interactive tool that enables to understand better smart meters time series data.
The application facilitate the understanding of electrical consumption patterns by identifying and localizing individual appliance usage within electrical consumption data.


## Contributors

* [Adrien Petralia](https://adrienpetralia.github.io/), EDF R&D, Université Paris Cité
* [Paul Boniol](https://boniolp.github.io/), Inria, ENS, PSL University, CNRS
* [Philippe Charpentier](https://www.researchgate.net/profile/Philippe-Charpentier), EDF R&D
* [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/), Université Paris Cité, IUF

## Usage

**Step 1:** Clone this repository using `git` and change into its root directory.

```(bash)
git clone https://github.com/boniolp/wichdevice.git
cd whichdevice/
```

**Step 2:** Create and activate a `conda` environment and install the dependencies.

```(bash)
conda create -n wichdevice python=3.8
conda activate wichdevice
pip install -r requirements.txt
```

**Step 3:** You can use our tool in two different ways: 

- Access online: https://wichdevice.streamlit.app/
- Run locally (preferable for faster interaction). To do so, run the following command:

```(bash)
streamlit run Hello.py
```

## Acknowledgments

Work supported by EDF R&D and ANRT French program.
