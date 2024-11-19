import setuptools
import os

VERSION = '1.1.0a'
DESCRIPTION = 'multiphaseflowsr'

#REQUIRED = open("requirements_pip.txt").read().splitlines()

#EXTRAS = {
#    "display": [
#        "pygraphviz",
#    ],
#}
#EXTRAS['all'] = list(set([item for group in EXTRAS.values() for item in group]))

PATH_UMF_CSVs = os.path.join("benchmark", "UmfDataset", "*.csv")
package_data = [PATH_UMF_CSVs]

setuptools.setup(
    name             = 'multiphaseflowsr',
    version          = VERSION,
    description      = DESCRIPTION,
    author           = 'Zhong Xiang',
    author_email     = 'xiangzhong1997@outlook.com',
    license          = 'MIT',
    packages         = setuptools.find_packages(),
    package_data     = {"multiphaseflowsr": package_data},
    #install_requires = REQUIRED,
    #extras_require   = EXTRAS,
)