# -*- coding: utf-8 -*-
"""
Customised qutip.control stats for local Choi fid
"""

# started 2015 Oct 6
# this version 2018 April 6
# Authors: Ben Dive & Alexander Pitchford

import qutip.control.stats as stats

class StatsFidCompLocal(stats.Stats):
    """
    Collect additional processing stats from the ChoiLocal fidcomp
    """
    def clear(self):
        super(StatsFidCompLocal, self).clear()

        self.wall_time_pseudo_fids_compute = 0.0
        self.wall_time_total_fid_compute = 0.0
        self.wall_time_local_target_compute = 0.0
        self.wall_time_local_fid_grad_compute = 0.0

    def report_timings(self):
        super(StatsFidCompLocal, self).report_timings()
        tot = self.wall_time_optim
        print("** Custom fidelity timings **")
        print("Wall time computing pseudo fids: " +
              self._format_datetime(self.wall_time_pseudo_fids_compute, tot))
        print("Wall time computing total fid: " +
              self._format_datetime(self.wall_time_total_fid_compute, tot))
        print("Wall time computing local targets: " +
              self._format_datetime(self.wall_time_local_target_compute, tot))
        print("Wall time computing local fid grad: " +
              self._format_datetime(self.wall_time_local_fid_grad_compute, tot))
        print("")
