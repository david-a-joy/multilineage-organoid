#!/usr/bin/env python

# Stdlib
import unittest

# 3rd party
import numpy as np

# Our own imports
from cm_microtissue_func import signals

# Tests


class TestCalcStatsAroundPeak(unittest.TestCase):

    def test_stats_for_single_flat_line(self):

        time = np.linspace(0, 2*np.pi, 100)
        signal = np.zeros_like(time)

        peaks = (0, 25, 100)

        res = signals.calc_stats_around_peak(time, signal, peaks)
        exp = {
            'peak_value': 0.0,
            'peak_index': 25,
            'peak_start_index': 25,
            'peak_end_index': 25,
            'total_wave_time': 0.0,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_stats_for_single_line_up(self):

        time = np.linspace(0, 2*np.pi, 100)
        signal = time * 0.5

        peaks = (0, 25, 100)

        res = signals.calc_stats_around_peak(time, signal, peaks)
        exp = {
            'peak_value': 0.79,
            'peak_index': 25,
            'peak_start_index': 1,
            'peak_end_index': 25,
            'total_wave_time': 1.52,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_stats_for_single_line_down(self):

        time = np.linspace(0, 2*np.pi, 100)
        signal = time * -0.5

        peaks = (0, 25, 100)

        res = signals.calc_stats_around_peak(time, signal, peaks)
        exp = {
            'peak_value': -0.79,
            'peak_index': 25,
            'peak_start_index': 25,
            'peak_end_index': 96,
            'total_wave_time': 4.51,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_stats_for_sawtooth(self):

        time = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        signal = np.array([0, 0, 2, 4, 3, 2, 1, 0])

        peaks = (0, 3, 7)

        res = signals.calc_stats_around_peak(time, signal, peaks)
        exp = {
            'peak_value': 4,
            'peak_index': 3,
            'peak_start_index': 1,
            'peak_end_index': 7,
            'total_wave_time': 6,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_stats_for_single_peak(self):

        time = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(time)

        peaks = (0, 25, 100)

        res = signals.calc_stats_around_peak(time, signal, peaks)
        exp = {
            'peak_value': 1.0,
            'peak_index': 25,
            'peak_start_index': 0,
            'peak_end_index': 68,
            'total_wave_time': 4.32,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_stats_for_double_peak(self):

        time = np.linspace(0, 4*np.pi, 200)
        signal = np.sin(time)

        peaks = (25, 125, 200)

        res = signals.calc_stats_around_peak(time, signal, peaks)
        exp = {
            'peak_value': 1.0,
            'peak_index': 125,
            'peak_start_index': 81,
            'peak_end_index': 167,
            'total_wave_time': 5.43,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_stats_for_double_peak_offset(self):

        time = np.linspace(0, 4*np.pi, 200)
        signal = np.sin(time) + 4.0

        peaks = (25, 125, 200)

        res = signals.calc_stats_around_peak(time, signal, peaks)
        exp = {
            'peak_value': 5.0,
            'peak_index': 125,
            'peak_start_index': 81,
            'peak_end_index': 167,
            'total_wave_time': 5.43,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key


class TestRefineSignalPeaks(unittest.TestCase):

    def test_refines_empty_list(self):

        time = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(time)

        res = signals.refine_signal_peaks(time, signal, [])
        assert res == []

    def test_refines_single_peak(self):

        time = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(time)

        res = signals.refine_signal_peaks(time, signal, [25])
        assert len(res) == 1
        res = res[0]
        exp = {
            'peak_value': 1.0,
            'peak_index': 25,
            'peak_start_index': 0,
            'peak_end_index': 68,
            'total_wave_time': 4.32,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_refines_single_peak_bad_annotation(self):

        time = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(time)

        res = signals.refine_signal_peaks(time, signal, [25, 50])
        assert len(res) == 1
        res = res[0]
        exp = {
            'peak_value': 1.0,
            'peak_index': 25,
            'peak_start_index': 0,
            'peak_end_index': 68,
            'total_wave_time': 4.32,
        }
        for key, val in exp.items():
            assert round(res[key], 2) == round(exp[key], 2), key

    def test_refines_multiple_peaks_bad_annotations(self):

        time = np.linspace(0, 4*np.pi, 200)
        signal = np.sin(time)

        res = signals.refine_signal_peaks(time, signal, [25, 50, 75, 125, 150])
        exp = [
            {
                'peak_value': 1.0,
                'peak_index': 25,
                'peak_start_index': 0,
                'peak_end_index': 68,
                'total_wave_time': 4.29,
            },
            {
                'peak_value': 1.0,
                'peak_index': 125,
                'peak_start_index': 81,
                'peak_end_index': 167,
                'total_wave_time': 5.43,
            },
        ]
        assert len(res) == len(exp)
        for res_stats, exp_stats in zip(res, exp):
            for key, val in exp_stats.items():
                assert round(res_stats[key], 2) == round(exp_stats[key], 2), key

    def test_refines_multiple_peaks_bad_annotations_numpy_arrays(self):

        time = np.linspace(0, 4*np.pi, 200)
        signal = np.sin(time)

        res = signals.refine_signal_peaks(time, signal, [np.array([[25]]), 50, np.array([75]), 125, 150])
        exp = [
            {
                'peak_value': 1.0,
                'peak_index': 25,
                'peak_start_index': 0,
                'peak_end_index': 68,
                'total_wave_time': 4.29,
            },
            {
                'peak_value': 1.0,
                'peak_index': 125,
                'peak_start_index': 81,
                'peak_end_index': 167,
                'total_wave_time': 5.43,
            },
        ]
        assert len(res) == len(exp)
        for res_stats, exp_stats in zip(res, exp):
            for key, val in exp_stats.items():
                assert round(res_stats[key], 2) == round(exp_stats[key], 2), key


class TestCalcVelocityStats(unittest.TestCase):

    def test_not_enough_data(self):

        time = np.linspace(0, 10, 2)
        signal = 2 * time

        mean, std, max = signals.calc_velocity_stats(time, signal)

        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(std))
        self.assertTrue(np.isnan(max))

    def test_works_up(self):

        time = np.linspace(0, 10, 10)
        signal = 2 * time

        mean, std, max = signals.calc_velocity_stats(time, signal, direction='up', time_scale=1.0)

        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, 0.0)
        self.assertAlmostEqual(max, 2.0)

    def test_works_down(self):

        time = np.linspace(0, 10, 10)
        signal = -2 * time

        mean, std, max = signals.calc_velocity_stats(time, signal, direction='down', time_scale=1.0)

        self.assertAlmostEqual(mean, -2.0)
        self.assertAlmostEqual(std, 0.0)
        self.assertAlmostEqual(max, -2.0)

    def test_wrong_direction(self):

        time = np.linspace(0, 10, 10)
        signal = -2 * time

        mean, std, max = signals.calc_velocity_stats(time, signal, direction='up', time_scale=1.0)

        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(std))
        self.assertTrue(np.isnan(max))
