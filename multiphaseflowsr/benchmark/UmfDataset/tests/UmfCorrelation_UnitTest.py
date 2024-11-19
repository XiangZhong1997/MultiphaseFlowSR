import time
import unittest
import numpy as np
import torch

# Internal imports
import multiphaseflowsr.benchmark.UmfDataset.UmfCorrelation as Umf

class UmfCorrelationTest(unittest.TestCase):
    def test_loading_csv(self):
        # Loading type1 correlations
        try:
            type1_df = Umf. load_umf_type1_correlations_csv (Umf. PATH_UMF_TYPE1_CORRS_CSV)
        except:
            self.fail("Failed to load type1 correlations csv")
        # Loading type2 correlations
        try:
            type2_df = Umf. load_umf_type2_correlations_csv (Umf. PATH_UMF_TYPE2_CORRS_CSV)
        except:
            self.fail("Failed to load type2 correlations csv")
        # Loading type3 correlations
        try:
            type3_df = Umf. load_umf_type3_correlations_csv (Umf. PATH_UMF_TYPE3_CORRS_CSV)
        except:
            self.fail("Failed to load type3 correlations csv")
        # Loading type4 correlations
        try:
            type4_df = Umf. load_umf_type4_correlations_csv (Umf. PATH_UMF_TYPE4_CORRS_CSV)
        except:
            self.fail("Failed to load type4 correlations csv")

        assert len(type1_df) == 45, "Type1 correlations csv has wrong number of correlations."
        assert len(type2_df) == 43, "Type2 correlations csv has wrong number of correlations."
        assert len(type3_df) == 6, "Type3 correlations csv has wrong number of correlations."
        assert len(type4_df) == 18, "Type4 correlations csv has wrong number of correlations."

        assert np.array_equal(type1_df.columns, type2_df.columns, type3_df.columns, type4_df.columns), "Type1, type2, type3 and type4 correlations dfs have different columns"

        # Loading all correlations
        try:
            corrs_df = Umf.CORRS_UMF_DF
        except:
            self.fail("Failed to load all correlations csv")

        assert len(corrs_df) == 112, "All correlations df has wrong number of correlations."

        expected_columns = np.array(['Filename', 'Name', 'Set', 'Number', 'Output', 'Formula', '# variables',
                           'v1_name', 'v1_low', 'v1_high', 'v2_name', 'v2_low', 'v2_high',
                           'v3_name', 'v3_low', 'v3_high', 'v4_name', 'v4_low', 'v4_high',
                           'v5_name', 'v5_low', 'v5_high', 'v6_name', 'v6_low', 'v6_high',
                           'v7_name', 'v7_low', 'v7_high', 'v8_name', 'v8_low', 'v8_high',
                           'v9_name', 'v9_low', 'v9_high', 'v10_name', 'v10_low', 'v10_high'])
        assert np.array_equal(corrs_df.columns,expected_columns)

        return None

    def test_get_units(self):

        # Test length
        assert len(Umf.get_units("d_p")) == Umf.UMF_UNITS_VECTOR_SIZE, "Wrong length for units vector"
        # Test d_p
        assert np.array_equal(Umf.get_units('d_p'), np.array([1., 0, 0., 0., 0.])), "Failed to get units for d_p"
        # Test rho
        assert np.array_equal(Umf.get_units('rho'), np.array([-3., 0., 1., 0., 0.])), "Failed to get units for rho"
        # Test g
        assert np.array_equal(Umf.get_units('g'), np.array([1., -2., 0., 0., 0.])), "Failed to get units for g"
        # Test mu
        assert np.array_equal(Umf.get_units('mu'), np.array([-1., -1., 1., 0., 0.])), "Failed to get units for mu "

        return None

    def test_UmfCorrelation(self):

        # Test loading a Correlation
        try:
            relatpb = Umf.UmfCorrelation(corr_name ="Ergun")
        except:
            self.fail("Failed to load a correlation")
        assert relatpb.corr_name == "Ergun", "Wrong corr_name."

        try:
            relatpb = Umf.UmfCorrelation(0)
        except:
            self.fail("Failed to load a correlation")
        assert relatpb.corr_name == "Ergun", "Wrong corr_name."

        # Test variable names on sample correlation
        expected_original_var_names = ['mu', 'd_p', 'rho', 'rho_p', 'Ar', 'epsilon', 'phi', 'g',]
        expected_original_y_name    = 'U_mf'
        expected_standard_var_names = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']
        expected_standard_y_name    = 'y'
        relatpb = Umf.UmfCorrelation(0, original_var_names = True) # With original var names
        assert np.array_equal(relatpb.X_names,          expected_original_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.X_names_original, expected_original_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.y_name,          expected_original_y_name), "Wrong y_name."
        assert np.array_equal(relatpb.y_name_original, expected_original_y_name), "Wrong y_name."
        assert relatpb.n_vars == 8, "Wrong n_vars."

        relatpb = Umf.UmfCorrelation(0, original_var_names = False) # Without original var names
        assert np.array_equal(relatpb.X_names,          expected_standard_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.X_names_original, expected_original_var_names), "Wrong X_names."
        assert np.array_equal(relatpb.y_name,          expected_standard_y_name), "Wrong y_name."
        assert np.array_equal(relatpb.y_name_original, expected_original_y_name), "Wrong y_name."
        assert relatpb.n_vars == 8, "Wrong n_vars."

        # Test units on sample correlation
        relatpb = Umf.UmfCorrelation(0)
        assert np.array_equal(relatpb.X_units, np.array([[ -1., -1., 1., 0., 0.],
                                                         [ 1., 0., 0., 0., 0.],
                                                         [ -3., 0., 1., 0., 0.],
                                                         [ -3., 0., 1., 0., 0.],
                                                         [ 0., 0., 0., 0., 0.],
                                                         [ 0., 0., 0., 0., 0.],
                                                         [ 0., 0., 0., 0., 0.],
                                                         [ 1., -2., 0., 0., 0.]])), "Wrong X_units."
        assert np.array_equal(relatpb.y_units, np.array([1., -1., 0., 0., 0.])), "Wrong y_units."

    def test_UmfCorrelation_datagen_all(self):
        verbose = False

        # Iterating through all Umf correlations (ie. correlations)
        for i in range(Umf.N_CORRS):

            # Loading correlation
            original_var_names = False  # replacing original symbols (e.g. mu, d_p etc.) by x0, x1 etc.
            # original_var_names = True  # using original symbols (e.g. mu, d_p etc.)
            pb = Umf.UmfCorrelation(i, original_var_names=original_var_names)

            if verbose:
                print("\n------------------------ %i : %s ------------------------" % (pb.i_corr, pb.corr_name))
                print(pb)

                # Print expression with evaluated constants
                print("--- Expression with evaluated constants ---")
                print(pb.formula_sympy_eval)

                # Printing physical units of variables
                print("--- Units ---")
                print("X units : \n", pb.X_units)
                print("y units : \n", pb.y_units)

            # Loading data sample
            X_array, y_array = pb.generate_data_points(n_samples=100)

            # Printing min, max of data points and warning if absolute value is above WARN_LIM
            if verbose: print("--- min, max ---")
            WARN_LIM = 50
            xmin, xmax, ymin, ymax = X_array.min(), X_array.max(), y_array.min(), y_array.max()
            if verbose:
                print("X min = ", xmin)
                print("X max = ", xmax)
                print("y min = ", ymin)
                print("y max = ", ymax)
                if abs(xmin) > WARN_LIM:
                    print("-> xmin has high absolute value : %f" % (xmin))
                if abs(xmax) > WARN_LIM:
                    print("-> xmax has high absolute value : %f" % (xmax))
                if abs(ymin) > WARN_LIM:
                    print("-> ymin has high absolute value : %f" % (ymin))
                if abs(ymax) > WARN_LIM:
                    print("-> ymax has high absolute value : %f" % (ymax))


if __name__ == '__main__':
    unittest.main()
