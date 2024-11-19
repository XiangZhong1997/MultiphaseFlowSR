import pandas as pd
import numpy as np
import pathlib
import sympy
import matplotlib.pyplot as plt

# Internal imports
from multiphaseflowsr.benchmark.utils import symbolic_utils as su

# Dataset paths
PARENT_FOLDER = pathlib.Path(__file__).parents[0]
PATH_UMF_TYPE1_CORRS_CSV       = PARENT_FOLDER / "UmfType1Correlations.csv"
PATH_UMF_TYPE2_CORRS_CSV       = PARENT_FOLDER / "UmfType2Correlations.csv"
PATH_UMF_TYPE3_CORRS_CSV       = PARENT_FOLDER / "UmfType3Correlations.csv"
PATH_UMF_TYPE4_CORRS_CSV       = PARENT_FOLDER / "UmfType4Correlations.csv"
PATH_UNITS_CSV             = PARENT_FOLDER / "units.csv"


# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- LOADING CSVs  ---------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def load_umf_type1_correlations_csv (filepath_type1_corrs ="UmfType1Correlations.csv"):
    """
    Loads UmfType1Correlations.csv into a clean pd.DataFrame .
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_corrs : str
        Path to UmfType1Correlations.csv.
    Returns
    -------
    corrs_umf_type1_df : pd.DataFrame
    """
    corrs_umf_type1_df = pd.read_csv(filepath_type1_corrs, sep=",")
    # drop last row(s) of NaNs
    corrs_umf_type1_df = corrs_umf_type1_df[~corrs_umf_type1_df[corrs_umf_type1_df.columns[0]].isnull()]
    # Set types for int columns
    corrs_umf_type1_df = corrs_umf_type1_df.astype({'Number': int, '# variables': int})
    # Number of correlations
    n_corrs = len(corrs_umf_type1_df)

    # ---- Verifying number of variables for safety ----
    # Checking the number of variables declared in the file for each problem
    # Expected number of variables for each problem
    expected_n_vars = (~corrs_umf_type1_df[["v%i_name" % (i) for i in range(1, 11)]].isnull().to_numpy()).sum(axis=1)       # (n_corrs,)
    # Declared number of variables for each problem
    n_vars = corrs_umf_type1_df["# variables"].to_numpy()                                                                   # (n_corrs,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                   # (n_corrs,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "correlations:\n %s"%(str(corrs_umf_type1_df.loc[~is_consistent]))


    # ---- Making four datasets consistent ----

    # Input variable related columns names: 'v1_name', 'v1_low', 'v1_high', 'v2_name' etc.
    variables_columns_names = np.array([['v%i_name'%(i), 'v%i_low'%(i), 'v%i_high'%(i)] for i in range (1,11)]).flatten()
    # Essential correlations related columns names: 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', etc.
    essential_columns_names = ['Output', 'Formula', '# variables'] + variables_columns_names.tolist()

    # Adding columns
    # Adding set columns indicating from which file these correlations come from (tpye1 file , type2 file, type3 file or type4 file)
    corrs_umf_type1_df["Set"] = "Type1"
    # Adding correlation names as a column (Ergun etc.)
    corrs_umf_type1_df["Name"] = corrs_umf_type1_df["Filename"]

    # Columns to keep: 'Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low',etc.
    columns_to_keep_names = ['Filename', 'Name', 'Set', 'Number'] + essential_columns_names
    # Selecting
    corrs_umf_type1_df = corrs_umf_type1_df[columns_to_keep_names]

    return corrs_umf_type1_df

def load_umf_type2_correlations_csv (filepath_type2_corrs ="UmfType2Correlations.csv"):
    """
    Loads UmfType2Correlations.csv into a clean pd.DataFrame .
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_corrs : str
        Path to UmfType2Correlations.csv.
    Returns
    -------
    corrs_umf_type2_df : pd.DataFrame
    """
    corrs_umf_type2_df = pd.read_csv(filepath_type2_corrs, sep=",")
    # drop last row(s) of NaNs
    corrs_umf_type2_df = corrs_umf_type2_df[~corrs_umf_type2_df[corrs_umf_type2_df.columns[0]].isnull()]
    # Set types for int columns
    corrs_umf_type2_df = corrs_umf_type2_df.astype({'Number': int, '# variables': int})
    # Number of correlations
    n_corrs = len(corrs_umf_type2_df)

    # ---- Verifying number of variables for safety ----
    # Checking the number of variables declared in the file for each problem
    # Expected number of variables for each problem
    expected_n_vars = (~corrs_umf_type2_df[["v%i_name" % (i) for i in range(1, 11)]].isnull().to_numpy()).sum(axis=1)       # (n_corrs,)
    # Declared number of variables for each problem
    n_vars = corrs_umf_type2_df["# variables"].to_numpy()                                                                   # (n_corrs,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                   # (n_corrs,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "correlations:\n %s"%(str(corrs_umf_type2_df.loc[~is_consistent]))


    # ---- Making four datasets consistent ----

    # Input variable related columns names: 'v1_name', 'v1_low', 'v1_high', 'v2_name' etc.
    variables_columns_names = np.array([['v%i_name'%(i), 'v%i_low'%(i), 'v%i_high'%(i)] for i in range (1,11)]).flatten()
    # Essential correlations related columns names: 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', etc.
    essential_columns_names = ['Output', 'Formula', '# variables'] + variables_columns_names.tolist()

    # Adding columns
    # Adding set columns indicating from which file these correlations come from (tpye1 file , type2 file, type3 file or type4 file)
    corrs_umf_type2_df["Set"] = "Type2"
    # Adding correlation names as a column (Ergun etc.)
    corrs_umf_type2_df["Name"] = corrs_umf_type2_df["Filename"]

    # Columns to keep: 'Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low',etc.
    columns_to_keep_names = ['Filename', 'Name', 'Set', 'Number'] + essential_columns_names
    # Selecting
    corrs_umf_type2_df = corrs_umf_type2_df[columns_to_keep_names]

    return corrs_umf_type2_df

def load_umf_type3_correlations_csv (filepath_type3_corrs ="UmfType3Correlations.csv"):
    """
    Loads UmfType3Correlations.csv into a clean pd.DataFrame .
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_corrs : str
        Path to UmfType3Correlations.csv.
    Returns
    -------
    corrs_umf_type3_df : pd.DataFrame
    """
    corrs_umf_type3_df = pd.read_csv(filepath_type3_corrs, sep=",")
    # drop last row(s) of NaNs
    corrs_umf_type3_df = corrs_umf_type3_df[~corrs_umf_type3_df[corrs_umf_type3_df.columns[0]].isnull()]
    # Set types for int columns
    corrs_umf_type3_df = corrs_umf_type3_df.astype({'Number': int, '# variables': int})
    # Number of correlations
    n_corrs = len(corrs_umf_type3_df)

    # ---- Verifying number of variables for safety ----
    # Checking the number of variables declared in the file for each problem
    # Expected number of variables for each problem
    expected_n_vars = (~corrs_umf_type3_df[["v%i_name" % (i) for i in range(1, 11)]].isnull().to_numpy()).sum(axis=1)       # (n_corrs,)
    # Declared number of variables for each problem
    n_vars = corrs_umf_type3_df["# variables"].to_numpy()                                                                   # (n_corrs,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                   # (n_corrs,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "correlations:\n %s"%(str(corrs_umf_type3_df.loc[~is_consistent]))


    # ---- Making four datasets consistent ----

    # Input variable related columns names: 'v1_name', 'v1_low', 'v1_high', 'v2_name' etc.
    variables_columns_names = np.array([['v%i_name'%(i), 'v%i_low'%(i), 'v%i_high'%(i)] for i in range (1,11)]).flatten()
    # Essential correlations related columns names: 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', etc.
    essential_columns_names = ['Output', 'Formula', '# variables'] + variables_columns_names.tolist()

    # Adding columns
    # Adding set columns indicating from which file these correlations come from (tpye1 file , type2 file, type3 file or type4 file)
    corrs_umf_type3_df["Set"] = "Type3"
    # Adding correlation names as a column (Ergun etc.)
    corrs_umf_type3_df["Name"] = corrs_umf_type3_df["Filename"]

    # Columns to keep: 'Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low',etc.
    columns_to_keep_names = ['Filename', 'Name', 'Set', 'Number'] + essential_columns_names
    # Selecting
    corrs_umf_type3_df = corrs_umf_type3_df[columns_to_keep_names]

    return corrs_umf_type3_df

def load_umf_type4_correlations_csv (filepath_type4_corrs ="UmfType4Correlations.csv"):
    """
    Loads UmfType4Correlations.csv into a clean pd.DataFrame .
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_corrs : str
        Path to UmfType4Correlations.csv.
    Returns
    -------
    corrs_umf_type4_df : pd.DataFrame
    """
    corrs_umf_type4_df = pd.read_csv(filepath_type4_corrs, sep=",")
    # drop last row(s) of NaNs
    corrs_umf_type4_df = corrs_umf_type4_df[~corrs_umf_type4_df[corrs_umf_type4_df.columns[0]].isnull()]
    # Set types for int columns
    corrs_umf_type4_df = corrs_umf_type4_df.astype({'Number': int, '# variables': int})
    # Number of correlations
    n_corrs = len(corrs_umf_type4_df)

    # ---- Verifying number of variables for safety ----
    # Checking the number of variables declared in the file for each problem
    # Expected number of variables for each problem
    expected_n_vars = (~corrs_umf_type4_df[["v%i_name" % (i) for i in range(1, 11)]].isnull().to_numpy()).sum(axis=1)       # (n_corrs,)
    # Declared number of variables for each problem
    n_vars = corrs_umf_type4_df["# variables"].to_numpy()                                                                   # (n_corrs,)
    # Is nb of declared variable consistent with variables columns ?
    is_consistent = np.equal(expected_n_vars, n_vars)                                                                   # (n_corrs,)
    assert is_consistent.all(), "Nb. of filled variables columns not consistent with declared nb. of variables for " \
                                "correlations:\n %s"%(str(corrs_umf_type4_df.loc[~is_consistent]))


    # ---- Making four datasets consistent ----

    # Input variable related columns names: 'v1_name', 'v1_low', 'v1_high', 'v2_name' etc.
    variables_columns_names = np.array([['v%i_name'%(i), 'v%i_low'%(i), 'v%i_high'%(i)] for i in range (1,11)]).flatten()
    # Essential correlations related columns names: 'Output', 'Formula', '# variables', 'v1_name', 'v1_low', etc.
    essential_columns_names = ['Output', 'Formula', '# variables'] + variables_columns_names.tolist()

    # Adding columns
    # Adding set columns indicating from which file these correlations come from (tpye1 file , type2 file, type3 file or type4 file)
    corrs_umf_type4_df["Set"] = "Type4"
    # Adding correlation names as a column (Ergun etc.)
    corrs_umf_type4_df["Name"] = corrs_umf_type4_df["Filename"]

    # Columns to keep: 'Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables', 'v1_name', 'v1_low',etc.
    columns_to_keep_names = ['Filename', 'Name', 'Set', 'Number'] + essential_columns_names
    # Selecting
    corrs_umf_type4_df = corrs_umf_type4_df[columns_to_keep_names]

    return corrs_umf_type4_df
    
def load_umf_all_correlations_csv (filepath_type1_corrs ="UmfType1Correlations.csv", filepath_type2_corrs ="UmfType2Correlations.csv", 
                                filepath_type3_corrs ="UmfType3Correlations.csv", filepath_type4_corrs ="UmfType4Correlations.csv", ):
    """
    Loads all UmfCorrelations.csv into a clean pd.DataFrame.
    Source files can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath_type1_corrs : str
        Path to UmfType1Correlations.csv.
    filepath_type2_corrs : str
        Path to UmfType2Correlations.csv.
    filepath_type3_corrs : str
        Path to UmfType3Correlations.csv.
    filepath_type4_corrs : str
        Path to UmfType4Correlations.csv.
    Returns
    -------
    corrs_umf_df : pd.DataFrame
    """
    type1_corrs_umf_df  = load_umf_type1_correlations_csv  (filepath_type1_corrs = filepath_type1_corrs)
    type2_corrs_umf_df  = load_umf_type2_correlations_csv  (filepath_type2_corrs = filepath_type2_corrs)
    type3_corrs_umf_df  = load_umf_type3_correlations_csv  (filepath_type3_corrs = filepath_type3_corrs)
    type4_corrs_umf_df  = load_umf_type4_correlations_csv  (filepath_type4_corrs = filepath_type4_corrs)

    corrs_umf_df = pd.concat((type1_corrs_umf_df, type2_corrs_umf_df, type3_corrs_umf_df, type4_corrs_umf_df),
                                # True so to get index going from 0 to 94 instead of 0 to 44 and then 45 to end
                               ignore_index=True)
    return corrs_umf_df


def load_umf_units_csv (filepath = "units.csv"):
    """
    Loads units.csv into a clean pd.DataFrame.
    Source file can be found here: https://space.mit.edu/home/tegmark/aifeynman.html
    Parameters
    ----------
    filepath : str
        Path to units.csv.
    Returns
    -------
    units_df : pd.DataFrame
    """
    units_df = pd.read_csv(filepath, sep=",")
    # drop last row(s) of NaNs
    units_df = units_df[~units_df[units_df.columns[0]].isnull()]
    # drop last column as it contains nothing
    units_df = units_df.iloc[:, :-1]

    return units_df

CORRS_UMF_DF = load_umf_all_correlations_csv (filepath_type1_corrs       = PATH_UMF_TYPE1_CORRS_CSV, 
                                         filepath_type2_corrs       = PATH_UMF_TYPE2_CORRS_CSV,
                                         filepath_type3_corrs       = PATH_UMF_TYPE3_CORRS_CSV, 
                                         filepath_type4_corrs       = PATH_UMF_TYPE4_CORRS_CSV,
                                                 )
UNITS_DF   = load_umf_units_csv     (filepath           = PATH_UNITS_CSV)

# Size of units vector
UMF_UNITS_VECTOR_SIZE = UNITS_DF.shape[1] - 2

# Number of correlations in dataset
N_CORRS = CORRS_UMF_DF.shape[0]

# ---------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------- UNITS UTILS  ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Gets units from variable name
def get_units (var_name):
    """
    Gets units of variable var_name. Example: get_units("mu")
    Parameters
    ----------
    var_name : str
    Returns
    -------
    units : numpy.array of shape (UMF_UNITS_VECTOR_SIZE,) of floats
        Units of variable.
    """
    assert not pd.isnull(var_name), "Can not get the units of %s as it is a null."%(var_name)
    try:
        units = UNITS_DF[UNITS_DF["Variable"] == var_name].to_numpy()[0][2:].astype(float)
    except:
        raise IndexError("Could not load units of %s"%(var_name))
    return units


# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- UMF CORRELATION  --------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
CONST_LOCAL_DICT = {"pi" : np.pi}

class UmfCorrelation:
    """
    Represents a single Umf benchmark problem.
    (See https://arxiv.org/abs/1905.11481 and https://space.mit.edu/home/tegmark/aifeynman.html for details).
    Attributes
    ----------
    i_corr : int
        Correlation number in the set of correlations (e.g. 0 to 44 for type1 corrs).
    corr_name : str
        Correlation name in the set of correlations (e.g. Ergun).
    n_vars : int
        Number of input variables.
    corr_df : pandas.core.series.Series
        Underlying pandas dataframe line of this correlation.
    original_var_names : bool
        Using original variable names (e.g. mu, phi etc.) and original output variable name (e.g. U_mf, Re_mf etc.) if
        True, using x0, x1 ... as input variable names and y as output variable name otherwise.

    y_name_original : str
        Name of output variable as in the Umf dataset.
    y_name : str
        Name of output variable.
    y_units : array_like of shape (UMF_UNITS_VECTOR_SIZE,) of floats
        Units of output variables.

    X_names_original : array_like of shape (n_vars,) of str
        Names of input variables as in the Umf dataset.
    X_names : array_like of shape (n_vars,) of str
        Names of input variables.
    X_lows : array_like of shape (n_vars,) of floats
        Lowest values taken by input variables.
    X_highs : array_like of shape (n_vars,) of floats
        Highest values taken by input variables.
    X_units :  array_like of shape (n_vars, UMF_UNITS_VECTOR_SIZE,) of floats
        Units of input variables.

    formula_original : str
        Formula as in the Umf dataset.
    X_sympy_symbols : array_like of shape (n_vars,) of sympy.Symbol
        Sympy symbols representing each input variables with assumptions (negative, positive etc.).
    sympy_X_symbols_dict : dict of {str : sympy.Symbol}
        Input variables names to sympy symbols (w assumptions), can be passed to sympy.parsing.sympy_parser.parse_expr
        as local_dict.
    local_dict : dict of {str : sympy.Symbol or float}
        Input variables names to sympy symbols (w assumptions) and constants (eg. pi : np.pi etc.), can be passed to
        sympy.parsing.sympy_parser.parse_expr as local_dict.
    formula_sympy : sympy expression
        Formula in sympy.
    formula_sympy_eval : sympy expression
        Formula in sympy with evaluated fixed constants (eg. pi -> 3.14... etc).
    formula_latex : str
        Formula in latex.
    """

    def __init__(self, i_corr = None, corr_name = None, original_var_names = False):
        """
        Loads a Umf correlation based on its number in the set or its correlation name
        Parameters
        ----------
        i_corr : int
            Correlation number in the whole set of correlations (0 to 44 for type1 corrs).
        corr_name : str
            Correlation name in the set of correlations (e.g. Ergun).
        original_var_names : bool
            Using original variable names (e.g. mu, phi etc.) and original output variable name (e.g. U_mf, Re_mf etc.) if
            True, using x0, x1 ... as input variable names and y as output variable name otherwise.
        """
        # Selecting correlation line in dataframe
        if i_corr is not None:
            self.corr_df  = CORRS_UMF_DF.iloc[i_corr]                                     # pandas.core.series.Series
        elif corr_name is not None:
            self.corr_df = CORRS_UMF_DF[CORRS_UMF_DF ["Name"] == corr_name ].iloc[0]    # pandas.core.series.Series
        else:
            raise ValueError("At least one of correlation number (i_corr) or correlation name (corr_name) should be specified to select a Umf correlation.")

        # Correlation number (0 to 44 for type1 corrs, 45 to 87 for type2 corrs, 88 to 93 for type3 corrs and 94 to 111 for type4 corrs)
        self.i_corr = i_corr                                                     # int
        # Correlation number in individual datasets (1 to 45 for type1 corrs, 1 to 43 for type2 corrs, 1 to 6 for type3 corrs and 1 to 18 for type4 corrs)
        self.i_corr_umf = self.corr_df["Number"]                                # str
        # Code name of correlation (eg. 'Ergun')
        self.corr_name = self.corr_df["Name"]                                    # str
        # Filename column in the Umf dataset
        self.corr_filename = self.corr_df["Filename"]
# str
        # Set column in the Umf dataset
        self.corr_set = self.corr_df["Set"]   
# str
        # SRBench style name
        self.SRBench_name = "umf_" + self.corr_filename  # str
        # Number of input variables
        self.n_vars = int(self.corr_df["# variables"])                         # int
        # Using x0, x1 ... and y names or original names (e.g. mu, phi, f etc.)
        self.original_var_names = original_var_names                         # bool

        # ----------- y : output variable -----------
        # Name of output variable
        self.y_name_original = self.corr_df["Output"]                              # str
        # Name of output variable : y or original name (eg. U_mf, Re_mf etc.)
        self.y_name = self.y_name_original if self.original_var_names else 'y'   # str
        # Units of output variables
        self.y_units = get_units(self.y_name)                                    # (UMF_UNITS_VECTOR_SIZE,)

        # ----------- X : input variables -----------
        # Utils id of input variables v1, v2 etc. in .csv
        var_ids_str = np.array( [ "v%i"%(i_var) for i_var in range(1, self.n_vars+1) ]   ).astype(str)                     # (n_vars,)
        # Names of input variables
        self.X_names_original = np.array( [ self.corr_df[ id + "_name" ] for id in var_ids_str  ]   ).astype(str)            # (n_vars,)
        X_names_xi_style      = np.array( [ "x%i"%(i_var) for i_var in range(self.n_vars)     ]   ).astype(str)            # (n_vars,)
        self.X_names          = self.X_names_original if self.original_var_names else X_names_xi_style                     # (n_vars,)
        # Lowest values taken by input variables
        self.X_lows           = np.array( [ self.corr_df[ id + "_low"  ] for id in var_ids_str ]    ).astype(float)          # (n_vars,)
        # Highest values taken by input variables
        self.X_highs          = np.array( [ self.corr_df[ id + "_high" ] for id in var_ids_str  ]   ).astype(float)          # (n_vars,)
        # Units of input variables
        self.X_units          = np.array( [ get_units(self.corr_df[ id + "_name" ]) for id in var_ids_str ] ).astype(float)  # (n_vars, UMF_UNITS_VECTOR_SIZE,)

        # ----------- Formula -----------
        self.formula_original = self.corr_df["Formula"] # (str)

        # Input variables as sympy symbols
        self.X_sympy_symbols = []
        for i in range(self.n_vars):
            is_positive = self.X_lows  [i] > 0
            is_negative = self.X_highs [i] < 0
            # If 0 is in interval, do not give assumptions as otherwise sympy will assume 0
            if (not is_positive) and (not is_negative):
                is_positive, is_negative = None, None
            self.X_sympy_symbols.append(sympy.Symbol(self.X_names[i],                                                   #  (n_vars,)
                                                     # Useful assumptions for simplifying etc
                                                     real     = True,
                                                     positive = is_positive,
                                                     negative = is_negative,
                                                     # If nonzero = False assumes that always = 0 which causes problems
                                                     # when simplifying
                                                     # nonzero  = not (self.X_lows[i] <= 0 and self.X_highs[i] >= 0),
                                                     domain   = sympy.sets.sets.Interval(self.X_lows[i], self.X_highs[i]),
                                                     ))

        # Input variables names to sympy symbols dict
        self.sympy_X_symbols_dict = {self.X_names[i] : self.X_sympy_symbols[i] for i in range(self.n_vars)}                     #  (n_vars,)
        # Dict to use to read original umf dataset formula
        # Original names to symbols in usage (i.e. symbols having original names or not)
        # eg. 'mu' -> mu symbol etc. (if original_var_names=True) or 'mu' -> x0 symbol etc. (else)
        self.sympy_original_to_X_symbols_dict = {self.X_names_original[i] : self.X_sympy_symbols[i] for i in range(self.n_vars)} #  (n_vars,)
        # NB: if original_var_names=True, then self.sympy_X_symbols_dict = self.sympy_original_to_X_symbols_dict

        # Declaring input variables via local_dict to avoid confusion
        # Eg. So sympy knows that we are referring to rho as a variable and not the function etc.
        # evaluate = False avoids eg. sin(mu) = 0 when mu domain = [0,5] ie. nonzero=False, but no need for this
        # if nonzero assumption is not used
        evaluate = False
        self.formula_sympy   = sympy.parsing.sympy_parser.parse_expr(self.formula_original,
                                                                     local_dict = self.sympy_original_to_X_symbols_dict,
                                                                     evaluate   = evaluate)
        # Local dict : dict of input variables (sympy_original_to_X_symbols_dict) and fixed constants (pi -> 3.14.. etc)
        self.local_dict = {}
        self.local_dict.update(self.sympy_original_to_X_symbols_dict)
        self.local_dict.update(CONST_LOCAL_DICT)
        self.formula_sympy_eval = sympy.parsing.sympy_parser.parse_expr(self.formula_original,
                                                                     local_dict = self.local_dict,
                                                                     evaluate   = evaluate)
        # Latex formula
        self.formula_latex   = sympy.printing.latex(self.formula_sympy)
        return None

    def target_function(self, X):
        """
        Evaluates X with target function.
        Parameters
        ----------
        X : numpy.array of shape (n_vars, ?,) of floats
        Returns
        -------
        y : numpy.array of shape (?,) of floats
        """
        # Getting sympy function
        f = sympy.lambdify(self.X_sympy_symbols, self.formula_sympy, "numpy")
        # Mapping between variables names and their data value
        mapping_var_name_to_X = {self.X_names[i]: X[i] for i in range(len(self.X_names))}
        # Evaluation
        # Forcing float type so if some symbols are not evaluated as floats (eg. if some variables are not declared
        # properly in source file) resulting partly symbolic expressions will not be able to be converted to floats
        # and an error can be raised).
        # This is also useful for detecting issues such as sin(mu) = 0 because theta.is_nonzero = False -> the result
        # is just an int of float
        y = f(**mapping_var_name_to_X)
        # Check if y is a sympy object, if so convert it to a numpy array
        if isinstance(y, sympy.Basic):
            y = np.array([float(yi) for yi in y])
        return y

    def generate_data_points(self, n_samples=1000):
        """
        Generates data points for this Umf correlation, specifically addressing the computation of the composite variable 'Ar'.
        This function dynamically computes 'Ar' using specific variables from the dataset, integrates it into the data points,
        and calculates the target function's output.
    
        Parameters
        ----------
        n_samples : int
            Number of samples to draw. The default value is 1,000, which aligns with the typical dataset size used in 
            AI Feynman challenges (https://space.mit.edu/home/tegmark/aifeynman.html). Note that the SRBench benchmark 
            (https://arxiv.org/abs/2107.14351) typically uses 100,000 samples.
    
        Returns
        -------
        X : numpy.array
            A numpy array of shape (n_vars + 1, n_samples) of floats, where n_vars is the number of variables including the 
            dynamically calculated 'Ar'.
    
        y : numpy.array
            A numpy array of shape (n_samples,) of floats representing the output of the target function evaluated at the data points.
        """
        # Ensure self.X_names is a list
        if not isinstance(self.X_names, list):
            self.X_names = self.X_names.tolist()
        
        # Define derived variables and their formulas
        derived_vars = {
            'Ar': lambda X: (X[self.X_names.index('g')] * X[self.X_names.index('d_p')]**3 * 
                             X[self.X_names.index('rho')] * (X[self.X_names.index('rho_p')] - X[self.X_names.index('rho')])) / 
                             X[self.X_names.index('mu')]**2,
        # Add more derived variables and their formulas here as needed
          }

        # Identify non-derived variables
        non_derived_indices = [i for i in range(len(self.X_names)) if self.X_names[i] not in derived_vars]
    
        # Generate random data for non-derived variables
        X_non_derived = np.array([
            np.random.uniform(
                self.X_lows[i],
                self.X_highs[i],
                n_samples
            )
            for i in non_derived_indices
        ])

        # Initialize the full X array with placeholders for derived variables
        X = np.zeros((len(self.X_names), n_samples))

        # Insert non-derived variable data into the full X array
        for i, index in enumerate(non_derived_indices):
            X[index] = X_non_derived[i]

        # Calculate derived variables and insert into the full X array
        for var_name, formula in derived_vars.items():
            if var_name in self.X_names:  # Only calculate and insert if the derived variable is in X_names
                var_index = self.X_names.index(var_name)
                X[var_index] = formula(X)

        # Compute the target function y
        y = self.target_function(X)

        return X, y

    def show_sample(self, n_samples = 1000, do_show = True, save_path = None):
        X_array, y_array = self.generate_data_points(n_samples = n_samples)
        n_dim = X_array.shape[0]
        fig, ax = plt.subplots(n_dim, 1, figsize=(10, n_dim * 4))
        fig.suptitle(self.formula_original)
        for i in range(n_dim):
            curr_ax = ax if n_dim == 1 else ax[i]
            curr_ax.plot(X_array[i], y_array, 'k.', markersize=1.)
            curr_ax.set_xlabel("%s : %s" % (self.X_names[i], self.X_units[i]))
            curr_ax.set_ylabel("%s : %s" % (self.y_name    , self.y_units))
        if save_path is not None:
            fig.savefig(save_path)
        if do_show:
            plt.show()

    def compare_expression (self, trial_expr,
                            handle_trigo            = True,
                            prevent_zero_frac       = True,
                            prevent_inf_equivalence = True,
                            verbose=False):
        """
        Checks if trial_expr is symbolically equivalent to the target expression of this Umf problem, following a
        similar methodology as SRBench (see https://github.com/cavalab/srbench).
        I.e, it is deemed equivalent if:
            - the symbolic difference simplifies to 0
            - OR the symbolic difference is a constant
            - OR the symbolic ratio simplifies to a constant
        Parameters
        ----------
        trial_expr : Sympy Expression
            Trial sympy expression with evaluated numeric free constants and assumptions regarding variables
            (positivity etc.) encoded in expression.
        handle_trigo : bool
            Tries replacing floats by rationalized factors of pi and simplify with that.
        prevent_zero_frac : bool
            If fraction = 0 does not consider expression equivalent.
        prevent_inf_equivalence: bool
            If symbolic error or fraction is infinite does not consider expression equivalent.
        verbose : bool
            Verbose.
        Returns
        -------
        is_equivalent, report : bool, dict
            Is the expression equivalent, A dict containing details about the equivalence SRBench style.
        """

        # Cleaning target expression like SRBench
        target_expr = su.clean_sympy_expr(self.formula_sympy_eval)

        is_equivalent, report = su.compare_expression (trial_expr  = trial_expr,
                                                       target_expr = target_expr,
                                                       handle_trigo            = handle_trigo,
                                                       prevent_zero_frac       = prevent_zero_frac,
                                                       prevent_inf_equivalence = prevent_inf_equivalence,
                                                       verbose                 = verbose,)

        return is_equivalent, report

    def trial_function (self, trial_expr, X):
        """
        Evaluates X on a trial expression mapping X to input variables names in sympy.
        Parameters
        ----------
        trial_expr : Sympy Expression
            Trial sympy expression with evaluated numeric free constants and assumptions regarding variables
            (positivity etc.) encoded in expression.
        X : numpy.array of shape (n_vars, ?,) of floats
        Returns
        -------
        y : numpy.array of shape (?,) of floats
        """
        # Getting sympy function
        f = sympy.lambdify(self.X_sympy_symbols, trial_expr, "numpy")
        # Mapping between variables names and their data value
        mapping_var_name_to_X = {self.X_names[i]: X[i] for i in range(self.n_vars)}
        # Evaluation
        # Forcing float type so if some symbols are not evaluated as floats (eg. if some variables are not declared
        # properly in source file) resulting partly symbolic expressions will not be able to be converted to floats
        # and an error can be raised).
        # This is also useful for detecting issues such as sin(mu) = 0 because theta.is_nonzero = False -> the result
        # is just an int of float
        y = f(**mapping_var_name_to_X)
        # forcing float type only if result is not already a single float (can happen if expression is a constant)
        if not isinstance(y, float):
            y = y.astype(float)
        return y

    def __str__(self):
        return "%s : Author(s) : %s\n%s"%(self.corr_set, self.corr_name, str(self.formula_sympy))

    def __repr__(self):
        return str(self)



