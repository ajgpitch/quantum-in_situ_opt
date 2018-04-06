# -*- coding: utf-8 -*-
"""
Created: 2016-12-5

Alexander Pitchford

quantum self optimiser result classes
"""
import sys
import numpy as np
import datetime
import re

def parse_time_delta(s):
    """Create timedelta object representing time delta
       expressed in a string

    Takes a string in the format produced by calling str() on
    a python timedelta object and returns a timedelta instance
    that would produce that string.

    Acceptable formats are:
        "HH:MM:SS"
        "X day, HH:MM:SS"
        "X days, HH:MM:SS"
        "HH:MM:SS.US"
        "X day, HH:MM:SS.US"
        "X days, HH:MM:SS.US"
    """
    if s is None:
        return None
    d = re.match(
            r'((?P<days>\d+) days?, )?(?P<hours>\d+):'
            r'(?P<minutes>\d+):(?P<seconds>\d+(\.\d+)?)',
            str(s)).groupdict(0)
    td = None
    try:
        td = datetime.timedelta(**dict(( (key, float(value))
                              for key, value in d.items() )))
    except:
        raise ValueError("Unable to parse {} as timedelta".format(s))

    return td

class Result(object):

    opt_file_attribs = [
        'evo_time', 'num_tslots',
        'fid_err_targ', 'numer_acc'
        ]
    attrib_fmts = {
        'evo_time':'0.3f', 'num_tslots':'d',
        'fid_err_targ':'0.4e', 'numer_acc':'0.6e'
        }

    def __init__(self):
        self.reset()

    def reset(self):
        self.avail_attrib_fmts = Result.attrib_fmts.copy()
        self.col_delim = '\t'
        self.list_delim = ';'
        self.num_tslots = None
        self.evo_time = None
        self.numer_acc = None

    def reset_file_attribs(self):
        self.file_attribs = list(self.def_file_attribs)
        self.file_attrib_fmts = [self.avail_attrib_fmts[a]
                                 for a in self.file_attribs]

    def add_opt_file_attribs(self, force_all=False):
        for a in self.opt_file_attribs:
            if force_all or getattr(self, a) is not None:
                self._add_file_attrib(a)

    def _add_file_attrib(self, a, to_start=True):
        if a not in self.file_attribs:
            if to_start:
                self.file_attribs = [a] + self.file_attribs
                self.file_attrib_fmts = [self.avail_attrib_fmts[a]] + \
                                                self.file_attrib_fmts
            else:
                self.file_attribs.append(a)
                self.file_attrib_fmts.append(self.avail_attrib_fmts[a])


    def write_file_header(self, f):
        l = ""
        for a in self.file_attribs:
            l += "{}\t".format(a)
        l = l[:-1] + "\n"
        f.write(l)

    def write_line(self, f):
        l = ""
        for a in zip(self.file_attribs, self.file_attrib_fmts):
            val = getattr(self, a[0])
            #print("Writing attrib: {}={}, fmt: {}".format(
            #                                  a[0], val, a[1]))
            if isinstance(val, list):
                lvs = ""
                for lv in val:
                    lvs += format(lv, a[1]) + ";"
                if ";" in lvs:
                    lvs = lvs[:-1]
                l += lvs + "\t"
            elif a[1].lower() == 'timedelta':
                l += str(datetime.timedelta(seconds=val)) + "\t"
            else:
                #print("val {}, format {}".format(val, a[1]))
                l += format(val, a[1]) + "\t"
        l = l[:-1] + "\n"
        f.write(l)

    @classmethod
    def load_from_txt(cls, f, has_header=True):
        """Load results from the passed text file
        Return as a list
        """
        closef = False
        if not hasattr(f, 'readline'):
            f = open(f, 'r')
            closef = True

        results = []
        read_header = has_header
        first_line = True
        for l in f:
            if first_line:
                header = cls._create_blank()
                if read_header:
                    header.file_attribs = []
                    header.file_attrib_fmts = []
                    vals = l.split(header.col_delim)
                    for v in vals:
                        v = v.strip()
                        fmt = header.avail_attrib_fmts.get(v, None)
                        if fmt is None:
                            raise ValueError(
                                    "Unknown colum header '{}'".format(v))
                        header._add_file_attrib(v, to_start=False)
                first_line = False
                #print("file attribs: {}".format(header.file_attribs))
                #print("file formats: {}".format(header.file_attrib_fmts))
                if len(header.file_attribs) != len(header.file_attrib_fmts):
                    raise RuntimeError("File attribs len not matching fmts")
                continue

            # Each one now is a data line
            res = cls._create_blank()
            vals = l.split(header.col_delim)
            if len(header.file_attribs) != len(vals):
                raise RuntimeError("Vals len not file attribs len")
            for fav in zip(header.file_attribs, header.file_attrib_fmts, vals):
                v = fav[2].strip()
                #print("Reading attrib: {}={}, fmt: {}".format(
                #                                  fav[0], v, fav[1]))
                if fav[1] == 'timedelta':
                    setattr(res, fav[0], parse_time_delta(v).total_seconds())
                elif isinstance(getattr(res, fav[0]), list):
                    # Should really format the slit objects
                    setattr(res, fav[0], v.split(res.list_delim))
                elif len(fav[1]) > 0:
                    if fav[1].endswith('d'):
                        setattr(res, fav[0], int(v))
                    elif fav[1].endswith(('f', 'g', 'e')):
                        setattr(res, fav[0], float(v))
                    else:
                        raise ValueError(
                                "No option for format '{}'".format(fav[1]))
                else:
                    setattr(res, fav[0], v)

            results.append(res)

        if closef:
            f.close()

        return results


class RepResult(Result):
    """
    Result of a single repetition
    """

    def_file_attribs = [
        'fid_err_primary', 'fid_err_secondary',
        'num_iter', 'run_time', 'termination'
        ]

    attrib_fmts = {
        'fid_err_primary':'0.9g', 'fid_err_secondary':'0.9g',
        'num_iter':'d', 'run_time':'timedelta', 'termination':''
        }

    def __init__(self):
        self.reset()

    def reset(self):
        Result.reset(self)
        self.avail_attrib_fmts.update(RepResult.attrib_fmts)
        self.reset_file_attribs()
        self.fid_err_primary = np.Inf
        self.fid_err_secondary = np.Inf
        self.num_iter = 0
        self.termination = ""
        self.run_time = 0.0
        self.local_search = False

    @property
    def formatted_run_time(self):
        return datetime.timedelta(seconds=self.run_time)

    def write(self, f):
        self.write_line(f)

class MultiRepResult(Result):
    """
    The results of the multiple repitition analyis
    """
    def_file_attribs = [
        'num_res', 'total_run_time',
        'min_primary_fid_err', 'num_primary_success',
        'avg_primary_success_run_time',
        'mean_iter_primary_success', 'std_iter_primary_success',
        'num_secondary_success',
        'mean_iter_secondary_success', 'std_iter_secondary_success',
        'mean_primary_success_second_fid', 'std_primary_success_second_fid',
        'termination'
        ]

    attrib_fmts = {
        'num_res':'d', 'total_run_time':'timedelta',
        'min_primary_fid_err':'0.9e', 'num_primary_success':'d',
        'avg_primary_success_run_time':'timedelta',
        'mean_iter_primary_success':'0.3f', 'std_iter_primary_success':'0.3f',
        'num_secondary_success':'d',
        'mean_iter_secondary_success':'0.3f',
        'std_iter_secondary_success':'0.3f',
        'mean_primary_success_second_fid':'0.9e',
        'std_primary_success_second_fid':'0.7e',
        'termination':''
        }

    def __init__(self, fid_err_targ, local_search,
                 num_tslots=None, evo_time=None, numer_acc=None):
        self.reset()
        self.local_search = local_search
        self.fid_err_targ = fid_err_targ
        self.num_tslots = num_tslots
        self.evo_time = evo_time
        self.numer_acc = numer_acc

    @classmethod
    def _create_blank(cls):
        return cls(0.0, False)

    def reset(self):
        Result.reset(self)
        self.avail_attrib_fmts.update(MultiRepResult.attrib_fmts)
        self.reset_file_attribs()
        self.results = []
        self.sorted = False
        self.local_search = False
        self.fid_err_targ = 0.0
        self.num_tslots = None
        self.evo_time = None
        self.numer_acc = None
        self.reset_analysis()

    def reset_analysis(self):
        self.num_res = len(self.results)
        self.total_run_time = 0.0
        self.total_primary_success_run_time = 0.0
        self.min_primary_fid_err = np.Inf
        self.avg_primary_success_run_time = 0.0
        self.num_primary_success = 0
        self.mean_iter_primary_success = 0.0
        self.std_iter_primary_success = 0.0
        self.mean_primary_success_second_fid = 0.0
        self.std_primary_success_second_fid = 0.0
        self.num_secondary_success = 0
        self.mean_iter_secondary_success = 0.0
        self.std_iter_secondary_success = 0.0
        self.termination = []

    def add_optim_result(self, optres):
        """
        Create a RepResult from the optimiser result and add to the list
        """
        fid_comp = optres.optimizer.dynamics.fid_computer
        repres = RepResult()
        repres.fid_err_primary = optres.fid_err
        repres.local_search = fid_comp.local
        if repres.local_search:
            repres.fid_err_secondary = 1 - fid_comp.compute_global_choi_fid()
        repres.num_iter = optres.num_iter
        repres.termination = optres.termination_reason
        repres.run_time = optres.wall_time
        repres.num_tslots = self.num_tslots
        repres.evo_time = self.evo_time
        repres.fid_err_targ = self.fid_err_targ
        repres.numer_acc = self.numer_acc
        repres.add_opt_file_attribs()

        self.results.append(repres)
        self.sorted = False

    def combine(self, other, sort=True):
        """ Combine this MultiRepResult with another"""
        self.add_results(other)
        self.num_res += other.num_res
        self.sort_results()
        self.combine_analysis(other)

    def add_results(self, other):
        """Add the results from another MultiRepResult"""
        self.results.extend(other.results)
        self.sorted = False

    def write_analysis(self, f, inc_header=False):
        if inc_header:
            self.write_file_header(f)
        self.write_line(f)

    def sort_results(self):
        sort_index = np.array([res.fid_err_primary
                                for res in self.results]).argsort()
        for i in sort_index:
            sorted_results = [self.results[i] for i in sort_index]
        self.results = sorted_results
        self.sorted = True
        self.num_res = len(self.results)

    def analyse_results(self):
        """
        Analyse the results of multiple repetitions
        """
        self.reset_analysis()
        #Sort the lists before output, will make things easier
        if not self.sorted: self.sort_results()

        """ Calculate some simple stats"""
        self.min_primary_fid_err = self.results[0].fid_err_primary
        #fid_err_primary = []
        #fid_err_secondary = []
        primary_success_iter = []
        #num_iter = []
        primary_success_second_fid = []
        secondary_success_iter = []
        for res in self.results:
            self.total_run_time += res.run_time
            if res.termination not in self.termination:
                self.termination.append(res.termination)

            if res.fid_err_primary <= self.fid_err_targ:
                self.num_primary_success += 1
                primary_success_iter.append(res.num_iter)
                self.total_primary_success_run_time += res.run_time
                if self.local_search:
                    primary_success_second_fid.append(res.fid_err_secondary)

            if (self.local_search and
                        res.fid_err_secondary <= self.fid_err_targ):
                self.num_secondary_success += 1
                secondary_success_iter.append(res.num_iter)

        if self.num_primary_success > 0:
            self.avg_primary_success_run_time = \
                self.total_primary_success_run_time / self.num_primary_success
            self.mean_iter_primary_success = np.mean(primary_success_iter)
            self.std_iter_primary_success = np.std(primary_success_iter)
        if self.local_search:
            if self.num_secondary_success > 0:
                self.mean_iter_secondary_success = np.mean(
                                                    secondary_success_iter)
                self.std_iter_secondary_success = np.std(
                                                    secondary_success_iter)
            if self.num_primary_success > 0:
                self.mean_primary_success_second_fid = np.mean(
                                                    primary_success_second_fid)
                self.std_primary_success_second_fid = np.std(
                                                    primary_success_second_fid)


    def _combine_mean_and_std(self, other, num_attrib, mean_attrib,
                              std_attrib=None):
        n1 = getattr(self, num_attrib)
        n2 = getattr(other, num_attrib)
        if n1==n2==0:
            #Nothing to combine
            return
        m1 = getattr(self, mean_attrib)
        m2 = getattr(other, mean_attrib)

        mc = (n1*m1 + n2*m2) / (n1 + n2)
        if std_attrib is not None:
            s1 = getattr(self, std_attrib)
            s2 = getattr(other, std_attrib)
            sc = np.sqrt((n1*s1**2 + n2*s2**2 + n1*(m1-mc)**2 + n2*(m2-mc)**2)
                                / (n1 + n2))
            setattr(self, std_attrib, sc)
        setattr(self, mean_attrib, mc)

    def _combine_mean(self, other, num_attrib, mean_attrib):
        self._combine_mean_and_std(other, num_attrib, mean_attrib)

    def combine_analysis(self, other):
        """Combine the analysis of these results with the 'other' result"""
        self.total_run_time += other.total_run_time
        self.total_primary_success_run_time += \
                        other.total_primary_success_run_time
        self.min_primary_fid_err = min(self.min_primary_fid_err,
                                       other.min_primary_fid_err)
        self._combine_mean(other, 'num_primary_success',
                                   'avg_primary_success_run_time')
        self._combine_mean_and_std(other, 'num_primary_success',
                                   'mean_iter_primary_success',
                                   'std_iter_primary_success')
        self._combine_mean_and_std(other, 'num_primary_success',
                                   'mean_primary_success_second_fid',
                                   'std_primary_success_second_fid')
        self._combine_mean_and_std(other, 'num_secondary_success',
                                   'mean_iter_secondary_success',
                                   'std_iter_secondary_success')
        self.num_primary_success += other.num_primary_success
        self.num_secondary_success += other.num_secondary_success
        self.termination.extend(tr for tr in other.termination
                                if tr not in self.termination)

    def report_analysis(self, f=sys.stdout):
        f.write("\n\nShort Summary:\n")
        f.write("Total number of successful repeats {}\n".format(self.num_res))
        f.write("Total run time {} HH:MM:SS.US\n".format(
                    datetime.timedelta(seconds=self.total_run_time)))

        f.write("\n{} runs had primary error below the threshold "
                "of {}\n".format(self.num_primary_success, self.fid_err_targ))
        if self.num_primary_success > 0:
            f.write("\twhere the average number of iterations "
                    "were {:0.4g} +/- {:0.3g}\n".format(
                            self.mean_iter_primary_success,
                           self.std_iter_primary_success))
            if self.local_search:
                f.write("\tand the average global fidelity error "
                        "was {:0.4g} +/- {:0.3g}\n".format(
                        self.mean_primary_success_second_fid,
                        self.std_primary_success_second_fid))
            f.write("and the avg run time was {} HH:MM:SS.US\n".format(
                    datetime.timedelta(
                            seconds=self.avg_primary_success_run_time)))

        if self.local_search:
            f.write("\n{} runs had secondary error below the threshold "
                    "of {}\n".format(self.num_secondary_success,
                                    self.fid_err_targ))
            if self.num_secondary_success > 0:
                f.write("\twhere the average number of iterations "
                        "were {:0.4g} +/- {:0.3g}\n".format(
                            self.mean_iter_secondary_success,
                            self.std_iter_secondary_success))
        if self.local_search:
            f.write("where Primary err is local choi, secondary is global choi\n")
        else:
            f.write("where Primary err is global choi\n")

    def write_results(self, f=sys.stdout,
                      inc_header=True, inc_opt_attribs=False):
        for res in self.results:
            #print("Writing result to {}. numer_acc={}".format(f,
            #                  res.numer_acc))
            if inc_opt_attribs:
                res.add_opt_file_attribs()
            if inc_header:
                res.write_file_header(f)
                inc_header = False
            res.write(f)

