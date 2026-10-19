## MuSR Beam Steering

Program to automatically steer beam onto sample in MuSR. Uses Mantid for data reduction and processing, a choice of 'metric functions' to quantify goodness of steering,
and a Generic Gaussian-Process + approximate Knowledge-Gradient Bayesian optimisation process to choose the next set of currents for steering magnets. 
This code has capabilities to be generalised to a wide range of settings where various PVs must be scanned over and a certain metric should be minimised.

This project relies on the <https://github.com/TymoteuszTula/PCA_Exp> repo. DOI: <https://doi.org/10.5281/zenodo.4495484>
Relevant files are found in ```musrbeamsteer/TT_PCA```.

Credit to Alex Louat for developing ```gp_kg_minimizer.py```.

## Dependencies and Usage

Would not yet recommend pip installing the package as dependencies are slightly complicated. Mantid must be installed in a mamba env (see <https://www.mantidproject.org/installation/index>) and genie_python pip installed into the same env. Other dependencies can be conda/pip installed at your preference.

Other dependencies:
- numpy
- scipy
- scikit-learn

To run code, naviagte into ```musrbeamsteer``` and execute ```optimised_integrated_gp.py```. Change settings as required under USER CONFIGURATION.

## Docs

Docs are a bit of a WIP but can be found at <https://isaac-baulk.github.io/musrbeamsteer/>.

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3) - see the [LICENSE](LICENSE) file for details. 
