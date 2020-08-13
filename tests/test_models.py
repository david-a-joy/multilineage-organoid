#!/usr/bin/env python3

# Stdlib
import unittest

# 3rd party
import numpy as np

# Our own imports
from multilineage_organoid import models

# Tests


class TestExpModelFit(unittest.TestCase):

    def test_fits_single_exp(self):

        x = np.linspace(0, 4.0, 100)
        y = 2.0*np.exp(-0.5*x) + 3.0

        fit = models.fit_model(x, y, x[1]-x[0])

        # Make sure we get the coefficients we plug in back
        coeffs = fit.get_single_exp_params()

        exp_coeffs = {
            'se_amp': 2.0,
            'se_offset': 3.0,
            'se_tc': 2.0,
        }

        self.assertEqual(coeffs.keys(), exp_coeffs.keys())
        for key in coeffs.keys():
            res = coeffs[key]
            exp = exp_coeffs[key]
            self.assertAlmostEqual(res, exp, msg=key)

        y_pred = fit.predict_single_exp_model()

        # Make sure the fit looks good
        r2 = np.mean((y-y_pred)**2)
        self.assertLess(r2, 1e-3)

    def test_fits_double_exp(self):

        x = np.linspace(0, 4.0, 10000)
        y = models.ExpModelFit.double_exp_model(
            x, amp=2.0, offset=3.0, tc1=2.0, tc2=4.0, tmean=0.5, tsigma=0.1)

        # Overdetermined, so don't check the coefficients for validity
        fit = models.fit_model(x, y, x[1]-x[0])

        y_pred = fit.predict_double_exp_model()

        # Make sure the fit looks good
        r2 = np.mean((y-y_pred)**2)
        self.assertLess(r2, 1e-3)
