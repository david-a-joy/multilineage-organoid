""" Tests for the I/O utilities """

# 3rd party
import numpy as np

# Our imports
import helpers
from cm_microtissue_func import io

# Tests


class TestDataReader(helpers.FileSystemTestCase):

    def test_read_ephys_invalid_level_shift(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write('1,0.1\n')
            fp.write('2,0.2\n')
            fp.write('3,0.3\n')

        reader = io.DataReader('ephys')
        with self.assertRaises(ValueError):
            reader.read_infile(infile, level_shift='bad')

    def test_read_ephys_no_header(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write('1,0.1\n')
            fp.write('2,0.2\n')
            fp.write('3,0.3\n')

        reader = io.DataReader('ephys')
        res_time, res_means = reader.read_infile(infile)
        exp_time = np.array([1, 2, 3])
        # This is due to level shifting, subtract mean, add 1
        exp_means = np.array([[1.0], [1.1], [1.2]])

        np.testing.assert_almost_equal(res_time, exp_time)
        np.testing.assert_almost_equal(res_means, exp_means)

    def test_read_ephys_with_header_range_shift(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write('Time,Blargh\n')
            fp.write('1,0.1\n')
            fp.write('2,0.2\n')
            fp.write('3,0.3\n')

        reader = io.DataReader('ephys')
        res_time, res_means = reader.read_infile(infile, level_shift='range')
        exp_time = np.array([1, 2, 3])
        # This is due to level shifting, subtract mean, add 1
        exp_means = np.array([[1.0], [1.1], [1.2]])

        np.testing.assert_almost_equal(res_time, exp_time)
        np.testing.assert_almost_equal(res_means, exp_means)

    def test_read_ephys_with_header_zero_shift(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write('Time,Blargh\n')
            fp.write('1,0.1\n')
            fp.write('2,0.2\n')
            fp.write('3,0.3\n')

        reader = io.DataReader('ephys')
        res_time, res_means = reader.read_infile(infile, level_shift='zero')
        exp_time = np.array([1, 2, 3])
        # This is due to level shifting, subtract min, add 0.0
        exp_means = np.array([[0.0], [0.1], [0.2]])

        np.testing.assert_almost_equal(res_time, exp_time)
        np.testing.assert_almost_equal(res_means, exp_means)

    def test_read_ephys_with_header_one_shift(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write('Time,Blargh\n')
            fp.write('1,0.1\n')
            fp.write('2,0.2\n')
            fp.write('3,0.3\n')

        reader = io.DataReader('ephys')
        res_time, res_means = reader.read_infile(infile, level_shift='one')
        exp_time = np.array([1, 2, 3])
        # This is due to level shifting, subtract min, add 1.0
        exp_means = np.array([[1.0], [1.1], [1.2]])

        np.testing.assert_almost_equal(res_time, exp_time)
        np.testing.assert_almost_equal(res_means, exp_means)

    def test_read_ephys_with_header_min_shift(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write('Time,Blargh\n')
            fp.write('1,0.1\n')
            fp.write('2,0.2\n')
            fp.write('3,0.3\n')

        reader = io.DataReader('ephys')
        res_time, res_means = reader.read_infile(infile, level_shift='min')
        exp_time = np.array([1, 2, 3])
        # This is due to level shifting, subtract mean, add 1
        exp_means = np.array([[1.0], [1.1], [1.2]])

        np.testing.assert_almost_equal(res_time, exp_time)
        np.testing.assert_almost_equal(res_means, exp_means)

    def test_read_ephys_with_header_two_columns(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write('Time,Blargh,Blergh\n')
            fp.write('1,0.1,0.2\n')
            fp.write('2,0.2,0.4\n')
            fp.write('3,0.3,0.5\n')

        reader = io.DataReader('ephys')
        res_time, res_means = reader.read_infile(infile)
        exp_time = np.array([1, 2, 3])
        exp_means = np.array([[1.0, 1.0], [1.1, 1.2], [1.2, 1.3]])

        np.testing.assert_almost_equal(res_time, exp_time)
        np.testing.assert_almost_equal(res_means, exp_means)

    def test_read_ca_header_two_columns_with_insufficient_samples(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write(','.join([
                "﻿Time::Relative Time!!R",
                "Channel1_R1_Area::EGFP_R1_Area!!R",
                "Channel1_R1_IntensityMean::EGFP_R1_IntensityMean!!R",
                "Channel1_R1_IntensitySum1::EGFP_R1_IntensitySum1!!R",
                "Channel1_R1_IntensityMaximum::EGFP_R1_IntensityMaximum!!R",
            ])+'\n')
            fp.write('"Ms",,,,\n')

            # Write 9 samples
            fp.write('0,147347.72,690.1,15042286,836\n')
            fp.write('10.0075,147347.72,695.9,15170060,822\n')
            fp.write('20.0149,147347.72,694.5,15138143,841\n')
            fp.write('30.0224,147347.72,693.9,15126015,824\n')
            fp.write('40.0298,147347.72,692.6,15097348,853\n')
            fp.write('50.0373,147347.72,691.5,15072638,832\n')
            fp.write('60.0448,147347.72,690.0,15040568,842\n')
            fp.write('70.0522,147347.72,689.0,15020256,837\n')
            fp.write('80.0671,147347.72,687.1,14977789,821\n')

        reader = io.DataReader('ca')
        with self.assertRaises(OSError):
            reader.read_infile(infile)

        with infile.open('at') as fp:
            # Write the 10th sample
            fp.write('90.0597,147347.72,687.8,14993955,830\n')

        # Now reading should work
        res = reader.read_infile(infile)
        self.assertEqual(len(res), 2)

    def test_read_ca_header_two_columns_range_shift(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write(','.join([
                "﻿Time::Relative Time!!R",
                "Channel1_R1_Area::EGFP_R1_Area!!R",
                "Channel1_R1_IntensityMean::EGFP_R1_IntensityMean!!R",
                "Channel1_R1_IntensitySum1::EGFP_R1_IntensitySum1!!R",
                "Channel1_R1_IntensityMaximum::EGFP_R1_IntensityMaximum!!R",
            ])+'\n')
            fp.write('"Ms",,,,\n')
            fp.write('0,147347.72,690.1,15042286,836\n')
            fp.write('10.0075,147347.72,695.9,15170060,822\n')
            fp.write('20.0149,147347.72,694.5,15138143,841\n')
            fp.write('30.0224,147347.72,693.9,15126015,824\n')
            fp.write('40.0298,147347.72,692.6,15097348,853\n')
            fp.write('50.0373,147347.72,691.5,15072638,832\n')
            fp.write('60.0448,147347.72,690.0,15040568,842\n')
            fp.write('70.0522,147347.72,689.0,15020256,837\n')
            fp.write('80.0597,147347.72,687.8,14993955,830\n')
            fp.write('90.0671,147347.72,687.1,14977789,821\n')

        reader = io.DataReader('ca')
        res_time, res_means = reader.read_infile(infile, level_shift='range')
        exp_time = np.array([0, 10.0075, 20.0149, 30.0224, 40.0298, 50.0373, 60.0448, 70.0522, 80.0597, 90.0671])
        exp_means = np.array([
            [690.1],
            [695.9],
            [694.5],
            [693.9],
            [692.6],
            [691.5],
            [690.0],
            [689.0],
            [687.8],
            [687.1],
        ]) - 687.1 + 8.8  # Dynamic range of the signal + 1

        np.testing.assert_almost_equal(res_time, exp_time, decimal=2)
        np.testing.assert_almost_equal(res_means, exp_means, decimal=2)

    def test_read_ca_header_two_columns_min_shift(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write(','.join([
                "﻿Time::Relative Time!!R",
                "Channel1_R1_Area::EGFP_R1_Area!!R",
                "Channel1_R1_IntensityMean::EGFP_R1_IntensityMean!!R",
                "Channel1_R1_IntensitySum1::EGFP_R1_IntensitySum1!!R",
                "Channel1_R1_IntensityMaximum::EGFP_R1_IntensityMaximum!!R",
            ])+'\n')
            fp.write('"Ms",,,,\n')
            fp.write('0,147347.72,690.1,15042286,836\n')
            fp.write('10.0075,147347.72,695.9,15170060,822\n')
            fp.write('20.0149,147347.72,694.5,15138143,841\n')
            fp.write('30.0224,147347.72,693.9,15126015,824\n')
            fp.write('40.0298,147347.72,692.6,15097348,853\n')
            fp.write('50.0373,147347.72,691.5,15072638,832\n')
            fp.write('60.0448,147347.72,690.0,15040568,842\n')
            fp.write('70.0522,147347.72,689.0,15020256,837\n')
            fp.write('80.0597,147347.72,687.8,14993955,830\n')
            fp.write('90.0671,147347.72,687.1,14977789,821\n')

        reader = io.DataReader('ca')
        res_time, res_means = reader.read_infile(infile, level_shift='min')
        exp_time = np.array([0, 10.0075, 20.0149, 30.0224, 40.0298, 50.0373, 60.0448, 70.0522, 80.0597, 90.0671])
        exp_means = np.array([
            [690.1],
            [695.9],
            [694.5],
            [693.9],
            [692.6],
            [691.5],
            [690.0],
            [689.0],
            [687.8],
            [687.1],
        ])
        np.testing.assert_almost_equal(res_time, exp_time, decimal=2)
        np.testing.assert_almost_equal(res_means, exp_means, decimal=2)

    def test_read_ca_header_two_columns_missing_values(self):

        infile = self.tempdir / 'grr.csv'
        with infile.open('wt') as fp:
            fp.write(','.join([
                "﻿Time::Relative Time!!R",
                "Channel1_R1_Area::EGFP_R1_Area!!R",
                "Channel1_R1_IntensityMean::EGFP_R1_IntensityMean!!R",
                "Channel1_R1_IntensitySum1::EGFP_R1_IntensitySum1!!R",
                "Channel1_R1_IntensityMaximum::EGFP_R1_IntensityMaximum!!R",
            ])+'\n')
            fp.write('"Ms",,,,\n')
            fp.write('0,147347.72,690.1,15042286,836\n')
            fp.write('10.0,147347.72,695.9,15170060,822\n')
            fp.write('-1,147347.72,694.5,15138143,841\n')
            fp.write('30.0,147347.72,693.9,15126015,824\n')
            fp.write('40.0,147347.72,692.6,15097348,853\n')
            fp.write('50.0,-1,691.5,15072638,832\n')
            fp.write('60.0,147347.72,690.0,15040568,842\n')
            fp.write('70.0,147347.72,689.0,15020256,837\n')
            fp.write('80.0,147347.72,687.8,14993955,830\n')
            fp.write('90.0,147347.72,687.1,14977789,821\n')
            fp.write('100.0,147347.72,687.2,14977789,821\n')
            fp.write('110.0,147347.72,687.3,14977789,821\n')

        reader = io.DataReader('ca')
        res_time, res_means = reader.read_infile(infile, level_shift='min')
        exp_time = np.array([0, 10, -1, 30, 40, 50, 60, 70, 80, 90, 100, 110])
        exp_means = np.array([
            [690.1],
            [695.9],
            [np.nan],
            [693.9],
            [692.6],
            [np.nan],
            [690.0],
            [689.0],
            [687.8],
            [687.1],
            [687.2],
            [687.3],
        ])

        np.testing.assert_almost_equal(res_time, exp_time, decimal=2)
        np.testing.assert_almost_equal(res_means, exp_means, decimal=2)
