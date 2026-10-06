"""Visualization utilities used by the GeoDynFormer manuscript figures."""

import seaborn as sns
import pandas as pd
import numpy as np
from .taskdataset import AffectiveTaskStats
from .utils import Constants, spline_smooth


class PlotRTs(AffectiveTaskStats):
    """Plot RT distributions.

    Args
    ----
    stats_obj (AffectiveTaskStats instance): Data from the model/participant.
    """

    def __init__(self, stats_obj):
        self.__dict__ = stats_obj.__dict__
        self.palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']

    def plot_rt_dists(self, ax, plot_type):
        if plot_type == 'all':
            plot_df = self._format_all()
        elif plot_type == 'Trial_type_2':
            plot_df = self._format_by_switch(2)
        elif plot_type == 'Trial_type_3':
            plot_df = self._format_by_switch(3)
        else:
            raise ValueError("plot_type must be 'all', 'Trial_type_2', or 'Trial_type_3'")
        sns.violinplot(x='trial_type', y='rts', hue='model_or_user', data=plot_df, split=True, inner='quart', ax=ax,
                       palette=self.palette, cut=0, linewidth=0.5)
        if plot_type == 'all':
            ax.set_xticks([])
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('')
        ax.set_ylabel('')
        return ax

    def _format_as_df(self, plot_dists, model_or_user, trial_types):
        all_rts = pd.concat(plot_dists)
        m_u_array = []
        ttype_array = []
        for rts, mu, ttype in zip(plot_dists, model_or_user, trial_types):
            m_u_array.extend(len(rts) * [mu])
            ttype_array.extend(len(rts) * [ttype])
        plot_df = pd.DataFrame({'rts': all_rts, 'model_or_user': m_u_array, 'trial_type': ttype_array})
        return plot_df

    def _format_all(self):
        urts = self.df['urt_ms'][self.select(**{'ucorrect': 1})]
        mrts = self.df['mrt_ms'][self.select(**{'mcorrect': 1})]
        plot_dists = [urts, mrts]
        m_or_u = ['user', 'model']
        trial_types = ['N/A', 'N/A']
        return self._format_as_df(plot_dists, m_or_u, trial_types)

    def _format_by_switch(self, type):
        u_stay_inds = self.select(**{'ucorrect': 1, 'trial_type': type - 2})
        m_stay_inds = self.select(**{'mcorrect': 1, 'trial_type': type - 2})
        u_switch_inds = self.select(**{'ucorrect': 1, 'trial_type': type})
        m_switch_inds = self.select(**{'mcorrect': 1, 'trial_type': type})
        u_stay_rts = self.df['urt_ms'][u_stay_inds]
        m_stay_rts = self.df['mrt_ms'][m_stay_inds]
        u_switch_rts = self.df['urt_ms'][u_switch_inds]
        m_switch_rts = self.df['mrt_ms'][m_switch_inds]
        plot_dists = [u_stay_rts, u_switch_rts, m_stay_rts, m_switch_rts]
        if type == 2:
            trial_types = ['G2G', 'E2G', 'G2G', 'E2G']
        elif type == 3:
            trial_types = ['E2E', 'G2E', 'E2E', 'G2E']
        else:
            raise ValueError(f'type must be 2 or 3.')
        m_or_u = ['user', 'user', 'model', 'model']
        return self._format_as_df(plot_dists, m_or_u, trial_types)


class PlotModelLatents:
    """Plot the model latents in 3D.

    Args
    ----
    data (AffectiveTaskStats instance): Data to plot.
    post_on_dur (int, optional): Duration after stimulus onset to plot (ms).
    pcs_to_plot (list, optional): Which PCs to plot.
    fixed_points (pandas DataFrame, optional): Fixed points to plot.
    """
    default_colors = 2 * ['#377EB8', '#E41A1C', '#009E73', '#984EA3']

    def __init__(self, data, post_on_dur=1200, dims_to_plot=[0, 1, 2], fixed_points=None, plot_pre_onset=True):
        self.data = data
        self.dims_to_plot = dims_to_plot
        self.latents = data.windowed['umap_latents'][:, :, dims_to_plot]
        self.m_rts = data.df['mrt_ms'].to_numpy()
        self.step = data.step
        self.n_pre = data.n_pre
        self.t_off_ind = self.n_pre + 1 + np.round(post_on_dur / self.step).astype('int')
        if plot_pre_onset:
            self.t_on_ind = 0
        else:
            self.t_on_ind = self.n_pre
        self.fixed_points = fixed_points

    def plot_g2g_male_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        cue_vals = [(0, 0, 0)]
        labels = ['G2G - male-happy']
        styles = ['-']
        series = self._get_full_series_repeat(cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_g2g_male_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        cue_vals = [(0, 0, 1)]
        labels = ['G2G - male-angry']
        styles = ['-']
        series = self._get_full_series_repeat(cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_g2g_female_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        cue_vals = [(0, 1, 0)]
        labels = ['G2G - female-happy']
        styles = ['-']
        series = self._get_full_series_repeat(cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_g2g_female_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        cue_vals = [(0, 1, 1)]
        labels = ['G2G - female-angry']
        styles = ['-']
        series = self._get_full_series_repeat(cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_e2e_male_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(1, 0, 0)]
        labels = ['E2E - male-happy']
        styles = ['-']
        series = self._get_full_series_repeat(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return (ax, series)

    def plot_e2e_female_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(1, 1, 0)]
        labels = ['E2E - female-happy']
        styles = ['--']
        series = self._get_full_series_repeat(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return (ax, series)

    def plot_e2e_male_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, same_prev_stim=False, **kwargs):
        stim_cue_vals = [(1, 0, 1)]
        labels = ['E2E - male-angry']
        styles = ['-']
        series = self._get_full_series_repeat(stim_cue_vals, same_prev_stim=same_prev_stim)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return (ax, series)

    def plot_e2e_female_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, same_stim=False, **kwargs):
        stim_cue_vals = [(1, 1, 1)]
        labels = ['E2E - female-angry']
        styles = ['--']
        series = self._get_full_series_repeat(stim_cue_vals, same_prev_stim=same_stim)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return (ax, series)

    def plot_e2g_male_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, same_prev_stim=False, **kwargs):
        stim_cue_vals = [(0, 0, 1)]
        labels = ['E2G - male-angry']
        series = self._get_full_series_switch(stim_cue_vals, same_prev_stim=same_prev_stim)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_e2g_male_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(0, 0, 0)]
        labels = ['E2G - male-happy']
        series = self._get_full_series_switch(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_e2g_female_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, same_stim=False, **kwargs):
        stim_cue_vals = [(0, 1, 1)]
        labels = ['E2G - female-angry']
        series = self._get_full_series_switch(stim_cue_vals, same_prev_stim=same_stim)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_e2g_female_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(0, 1, 0)]
        labels = ['E2G - female-happy']
        series = self._get_full_series_switch(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_g2e_male_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(1, 0, 0)]
        labels = ['G2E - male-happy']
        styles = ['-']
        series = self._get_full_series_switch(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_g2e_male_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(1, 0, 1)]
        labels = ['G2E - male-angry']
        styles = ['-']
        series = self._get_full_series_switch(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_g2e_female_happy(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(1, 1, 0)]
        labels = ['G2E - female-happy']
        styles = ['-']
        series = self._get_full_series_switch(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_g2e_female_angry(self, ax, elev=30, azim=60, plot_task_centroid=False, **kwargs):
        stim_cue_vals = [(1, 1, 1)]
        labels = ['G2E - female-angry']
        styles = ['-']
        series = self._get_full_series_switch(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_full_conditions(self, ax, elev=30, azim=60, plot_task_centroid=False, is_switch=False, **kwargs):
        stim_cue_vals = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        if is_switch:
            labels = ['E2G - male-happy', 'E2G - male-angry', 'E2G - female-happy', 'E2G - female-angry',
                      'G2E - male-happy', 'G2E - male-angry', 'G2E - female-happy', 'G2E - female-angry']
        else:
            labels = ['G2G - male-happy', 'G2G - male-angry', 'G2G - female-happy', 'G2G - female-angry',
                      'E2E - male-happy', 'E2E - male-angry', 'E2E - female-happy', 'E2E - female-angry']
        if is_switch:
            series = self._get_full_series_switch(stim_cue_vals)
        else:
            series = self._get_full_series_repeat(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def plot_related_stim_conditions(self, ax, elev=30, azim=60, plot_task_centroid=False, is_switch=False, **kwargs):
        stim_cue_vals = [(0, 0), (0, 1), (1, 0), (1, 1)]
        if is_switch:
            labels = ['E2G - male', 'E2G - female', 'G2E - happy', 'G2E - angry']
            styles = ['--', '-.', '--', '-.']
        else:
            labels = ['G2G - male', 'G2G - female', 'E2E - happy', 'E2E - angry']
            styles = ['-', '--', '-', '--']
        if is_switch:
            series = self._get_task_series_switch(stim_cue_vals)
        else:
            series = self._get_task_series_repeat(stim_cue_vals)
        plot_kwargs = {'plot_series_onset': True, 'plot_series_rt': True, 'plot_task_centroid': plot_task_centroid,
                       'line_width': 0.5, 'line_styles': styles}
        plot_kwargs.update(kwargs)
        ax = self.plot_3d(series, labels, ax, elev=elev, azim=azim, **plot_kwargs)
        return ax

    def _get_full_series_repeat(self, stim_cue_vals, same_prev_stim=False):
        all_selections = []
        for this_stim_cue in stim_cue_vals:
            this_cue = this_stim_cue[0]
            this_stim_gender = this_stim_cue[1]
            this_stim_emotion = this_stim_cue[2]
            this_filters = {'stim_gender': this_stim_gender, 'stim_emotion': this_stim_emotion, 'task_cue': this_cue,
                            'prev_task_cue': this_cue}
            if same_prev_stim:
                this_filters['prev_stim_gender'] = this_stim_gender
                this_filters['prev_stim_emotion'] = this_stim_emotion
            this_inds = self.data.select(**this_filters)
            all_selections.append(this_inds)
        return all_selections

    def _get_full_series_switch(self, stim_cue_vals, same_prev_stim=False):
        all_selections = []
        for this_stim_cue in stim_cue_vals:
            this_cue = this_stim_cue[0]
            this_stim_gender = this_stim_cue[1]
            this_stim_emotion = this_stim_cue[2]
            this_filters = {'stim_gender': this_stim_gender, 'stim_emotion': this_stim_emotion, 'task_cue': this_cue,
                            'prev_task_cue': 1 - this_cue}
            if same_prev_stim:
                this_filters['prev_stim_gender'] = this_stim_gender
                this_filters['prev_stim_emotion'] = this_stim_emotion
            this_inds = self.data.select(**this_filters)
            all_selections.append(this_inds)
        return all_selections

    def _get_task_series_repeat(self, stim_cue_vals):
        all_selections = []
        for this_stim_cue in stim_cue_vals:
            this_cue = this_stim_cue[0]
            this_stim = this_stim_cue[1]
            if this_cue == 0:
                this_filters = {'stim_gender': this_stim, 'task_cue': this_cue, 'prev_task_cue': this_cue}
            else:
                this_filters = {'stim_emotion': this_stim, 'task_cue': this_cue, 'prev_task_cue': this_cue}
            this_inds = self.data.select(**this_filters)
            all_selections.append(this_inds)
        return all_selections

    def _get_task_series_switch(self, stim_cue_vals):
        all_selections = []
        for this_stim_cue in stim_cue_vals:
            this_cue = this_stim_cue[0]
            this_stim = this_stim_cue[1]
            if this_cue == 0:
                this_filters = {'stim_gender': this_stim, 'task_cue': this_cue, 'prev_task_cue': 1 - this_cue}
            else:
                this_filters = {'stim_emotion': this_stim, 'task_cue': this_cue, 'prev_task_cue': 1 - this_cue}
            this_inds = self.data.select(**this_filters)
            all_selections.append(this_inds)
        return all_selections

    def plot_3d(self, series, labels, ax, elev=30, azim=60, is_switch=False, **kwargs):
        R = kwargs.get('R', None)
        colors = kwargs.get('colors', self.default_colors)
        line_styles = kwargs.get('line_styles', len(series) * ['-'])
        width = kwargs.get('line_width', 2.0)
        if kwargs.get('plot_task_centroid', False):
            ax = self._plot_task_centroid(ax, is_switch=is_switch)
        plot_series_onset = kwargs.get('plot_series_onset', False)
        plot_series_rt = kwargs.get('plot_series_rt', False)
        plot_times = kwargs.get('plot_times', None)
        plot_markers = kwargs.get('markers', len(series) * [None])
        plot_t_posts = kwargs.get('plot_t_posts', len(series) * [1500])
        plot_line = kwargs.get('plot_line', True)
        for i, s in enumerate(series):
            label = labels[i]
            color = colors[i]
            style = line_styles[i]
            marker = plot_markers[i]
            t_post = plot_t_posts[i]
            if plot_line:
                ax = self._plot_3d_line(ax, s, color, style, label, width, R=R, marker=marker, t_post=t_post)
            if plot_series_onset:
                if not plot_line:
                    ax = self._mark_3d_plot(ax, s, self.n_pre, color=color, size=8, marker=marker, R=R)
                else:
                    ax = self._mark_3d_plot(ax, s, self.n_pre, 'k', 8, marker, R=R)
            if plot_series_rt:
                t_ind = self._get_series_rt_samples(s) + self.n_pre
                ax = self._mark_3d_plot(ax, s, t_ind, color, 8, 'o', R=R)
            if plot_times is not None:
                ax = self._plot_timepoints(ax, s, plot_times, color, R=R)
        if self.fixed_points is not None:
            ax = self._plot_fixed_points(ax, R=R)
        ax = self._adjust_plot(ax, elev, azim, **kwargs)
        return ax

    def _plot_timepoints(self, ax, series, times, color, R=None):
        for t in times:
            t_plot = int(self.n_pre + t / self.step)
            ax = self._mark_3d_plot(ax, series, t_plot, color, 7, 'x', R=R)
        return ax

    def _get_series_rt_samples(self, series):
        rt = np.round(np.mean(self.m_rts[series]) / self.step).astype('int')
        return rt

    def _plot_task_centroid(self, ax, is_switch=False):
        if is_switch:
            gender_color, gender_size, gender_marker = ('k', 20, 'o')
            emotion_color, emotion_size, emotion_marker = ('grey', 20, 'o')
        else:
            gender_color, gender_size, gender_marker = ('k', 20, '*')
            emotion_color, emotion_size, emotion_marker = ('grey', 20, '*')
        t_ind = self.n_pre
        if is_switch:
            gender_filter = {'task_cue': 0, 'prev_task_cue': 1}
            gender_inds = self.data.select(**gender_filter)
            for inds in [gender_inds]:
                ax = self._mark_3d_plot(ax, inds, t_ind, gender_color, gender_size, gender_marker, label='Task 1 (e2g)')
            emotion_filter = {'task_cue': 1, 'prev_task_cue': 0}
            emotion_inds = self.data.select(**emotion_filter)
            for inds in [emotion_inds]:
                ax = self._mark_3d_plot(ax, inds, t_ind, emotion_color, emotion_size, emotion_marker,
                                        label='Task 2 (g2e)')
        else:
            gender_filter = {'task_cue': 0, 'prev_task_cue': 0}
            gender_inds = self.data.select(**gender_filter)
            for inds in [gender_inds]:
                ax = self._mark_3d_plot(ax, inds, t_ind, gender_color, gender_size, gender_marker, label='Task 1 (g2g)')
            emotion_filter = {'task_cue': 1, 'prev_task_cue': 1}
            emotion_inds = self.data.select(**emotion_filter)
            for inds in [emotion_inds]:
                ax = self._mark_3d_plot(ax, inds, t_ind, emotion_color, emotion_size, emotion_marker,
                                        label='Task 2 (e2e)')
        return ax

    def _mark_3d_plot(self, ax, series_inds, t_ind, color, size, marker, label=None, R=None):
        latents = self.latents
        if R is not None:
            latents = latents @ R
        x = np.mean(latents[:, series_inds, 0], 1)
        y = np.mean(latents[:, series_inds, 1], 1)
        z = np.mean(latents[:, series_inds, 2], 1)
        ax.scatter(x[t_ind], y[t_ind], z[t_ind], marker=marker, facecolors=color, s=size, linewidth=0.5, label=label)
        return ax

    def _plot_3d_line(self, ax, series_inds, color, style, label, width, R=None, marker=None, t_post=None,
                      smooth=False):
        latents = self.latents
        if R is not None:
            latents = latents @ R
        if t_post is not None:
            t_off_ind = self.n_pre + 1 + np.round(t_post / self.step).astype('int')
        else:
            t_off_ind = self.t_off_ind
        mean_latents = np.mean(latents[self.t_on_ind:t_off_ind, series_inds, :], axis=1)
        if smooth is True:
            mean_latents = spline_smooth(mean_latents)
        x = mean_latents[:, 0]
        y = mean_latents[:, 1]
        z = mean_latents[:, 2]
        if marker is not None:
            ax.plot(x, y, z, color=color, label=label, linestyle=style, linewidth=width, marker=marker, markevery=20,
                    markersize=4, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
        else:
            ax.plot(x, y, z, color=color, label=label, linestyle=style, linewidth=width)
        return ax

    def _plot_fixed_points(self, ax, R=None):
        gender_fps = self.fixed_points.query('cue == 0')
        emotion_fps = self.fixed_points.query('cue == 1')
        plot_fps = [gender_fps, emotion_fps]
        fp_markers = ['x', 'x']
        colors = [Constants.COLOR_G2G, Constants.COLOR_E2E]
        size = 10
        zloc_key = 'zloc_umap'
        for plot_fp, mark, c in zip(plot_fps, fp_markers, colors):
            for fpz in plot_fp[zloc_key]:
                if R is not None:
                    fpz = fpz @ R
                ax.scatter(fpz[0], fpz[1], fpz[2], s=size, color=c, marker=mark, zorder=2, linewidth=0.5)
        return ax

    def _adjust_plot(self, ax, elev, azim, **kwargs):
        ax.view_init(elev=elev, azim=azim)
        ax.xaxis._axinfo['grid']['linewidth'] = 0.25
        ax.yaxis._axinfo['grid']['linewidth'] = 0.25
        ax.zaxis._axinfo['grid']['linewidth'] = 0.25
        ax.set_xlim(kwargs.get('xlim', None))
        ax.set_ylim(kwargs.get('ylim', None))
        ax.set_zlim(kwargs.get('zlim', None))
        if kwargs.get('remove_tick_labels', True):
            ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.set_zticklabels('')
        if kwargs.get('annotate', 'local'):
            ax = self._annotate_plot(ax, **kwargs)
        return ax

    def _annotate_plot(self, ax, **kwargs):
        axes_text = kwargs.get('annotate', 'local')
        if axes_text == 'local':
            ax.set_xlabel(f'UMAP {0 + 1}', labelpad=-15)
            ax.set_ylabel(f'UMAP {1 + 1}', labelpad=-15)
            ax.set_zlabel(f'UMAP{2 + 1}', labelpad=-15)
        elif axes_text == 'global':
            ax.set_xlabel(f'Task', labelpad=-15)
            ax.set_ylabel(f'Emotion', labelpad=-15)
            ax.set_zlabel(f'Gender', labelpad=-15)
        ax.set_title(kwargs.get('title', None))
        ax.legend(loc='upper center', ncol=2, frameon=False)
        return ax
