"""Shared utilities for GeoDynFormer training, analysis, and plotting."""

import itertools
import os

import numpy as np
import pandas as pd
import torch
import umap
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import pearsonr, special_ortho_group

_MARKER_MAP = {
    "o": "circle",
    "s": "square",
    "D": "diamond",
    "^": "triangle-up",
    "v": "triangle-down",
    "<": "triangle-left",
    ">": "triangle-right",
    "x": "x",
    "+": "cross",
    "*": "star",
    "P": "cross",
    "X": "x",
    None: None,
    "": None,
    "None": None,
}

_DASH_MAP = {
    "-": "solid",
    "--": "dash",
    "-.": "dashdot",
    ":": "dot",
}


# -----------------------------------------------------------------------------
# Figure export and plotting helpers
# -----------------------------------------------------------------------------

def mpl3d_fig_to_plotly_fig(fig, title=None, include_scatters=True):
    """
    Convert a Matplotlib Figure that may contain multiple Axes3D into a Plotly Figure
    with multiple 3D scenes (scene, scene2, ...).

    Preserves:
      - ax.lines (3D) as Scatter3d (lines / lines+markers)
      - (optional) ax.collections (3D scatter) as Scatter3d (markers)

    Note: This is a *rebuild* conversion (robust for 3D), not tls.mpl_to_plotly.
    """
    pfig = go.Figure()
    axes3d = []
    for ax in fig.axes:
        if hasattr(ax, 'get_zlabel') and hasattr(ax, 'get_proj'):
            axes3d.append(ax)
    if len(axes3d) == 0:
        raise ValueError('No 3D axes found in fig. (Expected mplot3d Axes3D.)')

    def _scene_id(i):
        return 'scene' if i == 0 else f'scene{i + 1}'

    for i, ax in enumerate(axes3d):
        scene_name = _scene_id(i)
        for line in getattr(ax, 'lines', []):
            if not hasattr(line, '_verts3d'):
                continue
            x, y, z = line._verts3d
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            z = np.asarray(z, float)
            color = line.get_color()
            lw = float(line.get_linewidth())
            ls = line.get_linestyle()
            dash = _DASH_MAP.get(ls, 'solid')
            marker = line.get_marker()
            ms = float(line.get_markersize())
            symbol = _MARKER_MAP.get(marker, None)
            mode = 'lines' if symbol is None else 'lines+markers'
            name = line.get_label()
            if not name or str(name).startswith('_'):
                name = None
            pfig.add_trace(
                go.Scatter3d(x=x, y=y, z=z, mode=mode, name=name, line=dict(color=color, width=max(4.0, lw), dash=dash),
                             marker=None if symbol is None else dict(symbol=symbol, size=max(20.0, ms),
                                                                     line=dict(width=1)), scene=scene_name))
        if include_scatters:
            for coll in getattr(ax, 'collections', []):
                if not hasattr(coll, '_offsets3d'):
                    continue
                xs, ys, zs = coll._offsets3d
                xs = np.asarray(xs, float)
                ys = np.asarray(ys, float)
                zs = np.asarray(zs, float)
                if xs.size == 0:
                    continue
                fc = coll.get_facecolor()
                if fc is not None and len(fc) > 0:
                    r, g, b, a = fc[0]
                    color = f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a:.3f})'
                else:
                    color = 'rgba(0,0,0,1)'
                sizes = coll.get_sizes()
                if sizes is not None and len(sizes) > 0:
                    size = float(np.sqrt(sizes[0]))
                else:
                    size = 6.0
                pfig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', name=None,
                                            marker=dict(size=max(3.0, size), color=color), scene=scene_name,
                                            showlegend=False))
        scene_layout = dict(aspectmode='data', xaxis=dict(title=ax.get_xlabel()), yaxis=dict(title=ax.get_ylabel()),
                            zaxis=dict(title=ax.get_zlabel()))
        pfig.update_layout(**{scene_name: scene_layout})
    pfig.update_layout(title=title, margin=dict(l=0, r=0, t=40 if title else 10, b=0),
                       legend=dict(yanchor='top', y=0.98, xanchor='left', x=0.02, bgcolor='rgba(255,255,255,0.7)'))
    return pfig


def p_to_stars(p):
    """Convert p-value to significance stars, including 4-star option."""
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


# -----------------------------------------------------------------------------
# Shared constants and training/data utilities
# -----------------------------------------------------------------------------

class Constants(object):
    eta = 1e-06
    T_START = 'START'
    T_RT = 'RT'
    T_HALF_RT = 'HALF_RT'
    T_NDT = 'NDT'
    T_HALF_NDT = 'HALF_NDT'
    T_THIRD_RT = 'THIRD_RT'
    TRIAL_START = 5
    NDT_REPEAT = 35
    RT_SEG_REPEAT = 60
    RT_REPEAT = 95
    NDT_SWITCH = 55
    RT_SEG_SWITCH = 55
    RT_SWITCH = 120
    COLOR_TASK_ARROW = '#D17A22'
    COLOR_G2G = '#1F4E79'
    COLOR_E2E = '#B23A2F'
    COLOR_E2G = '#1B7F79'
    COLOR_G2E = '#C48A1D'
    COLOR_REPEAT = '#6A5ACD'
    COLOR_SWITCH = '#5A5A5A'


class CustomBatch:

    def __init__(self, batch):
        self.batch = torch.stack(batch, dim=1)

    def pin_memory(self):
        self.batch = self.batch.pin_memory()
        return self


def custom_collate(batch):
    """Custom function for constructing batches on calls to DataLoader."""
    return CustomBatch(batch)


class ConfigMixin:
    """Mixin class which enables overriding parameters in the config file.
    This is useful when one wants to test out different sets of
    model/data/training parameters without creating a new config file
    for each set.
    """
    _model_params = {'latent_dim', 'init_rnn_dim', 'init_hidden_dim', 'w_dim', 'encoder_mlp_hidden_dim',
                     'encoder_rnn_input_dim', 'encoder_rnn_hidden_dim', 'encoder_rnn_dropout', 'encoder_rnn_n_layers',
                     'encoder_combo_hidden_dim', 'decoder_hidden_dim', 'trans_dim', 'prior_dist', 'posterior_dist',
                     'likelihood_dist', 'likelihood_scale_param', 'model_type', 'dynamics_matrix_mult',
                     'dynamics_init_method'}
    _training_params = {'num_epochs', 'batch_size', 'train_frac', 'val_frac', 'test_frac', 'optim_alg', 'LR',
                        'weight_decay', 'clip_grads', 'clip_val', 'n_workers', 'rand_seed', 'learn_prior', 'objective',
                        'do_amsgrad', 'start_temp', 'cool_rate', 'temp_update_every', 'stop_patience', 'stop_min_epoch',
                        'stop_delta', 'stop_metric'}
    _data_params = {'input_dim', 'u_dim', 'nth_play_range', 'outlier_method', 'outlier_thresh', 'keep_every'}
    _transform_splits = {'train', 'val', 'test'}
    _transform_params = {'step_size', 'duration', 'noise_type', 'noise_sd', 'noise_corr_weight', 'start_times',
                         'data_augmentation_type', 'data_aug_kernel_bandwidth', 'aug_rt_sd', 'aug_resample_frac',
                         'upscale_mult', 'min_trials', 'post_resp_buffer', 'smoothing_type', 'kernel_sd',
                         'match_accuracy', 'rt_method', 'remap_rt', 'optimal_min_rt', 'optimal_kernel_width'}
    _experiment_params = {'split_indices', 'processed_save_dir', 'mode', 'logger_type', 'do_early_stopping',
                          'params_to_load', 'neptune_proj_name', 'expt_tags', 'log_save_dir'}

    def update_params(self, **kwargs):
        for key, val in kwargs.items():
            if key in self._model_params:
                self.config_params['model_params'][key] = val
            elif key in self._training_params:
                self.config_params['training_params'][key] = val
            elif key in self._data_params:
                self.config_params['data_params'][key] = val
            elif key in self._transform_splits:
                split_key = f'{key}_transform_kwargs'
                for tkey, tval in val.items():
                    if tkey in self._transform_params:
                        self.config_params['data_params'][split_key][tkey] = tval
                    else:
                        raise KeyError(f'Invalid config key: {tkey}')
            elif key in self._experiment_params:
                pass
            else:
                raise KeyError(f'Invalid config key: {key}')
        self._update_experiment_options(**kwargs)

    def _update_experiment_options(self, **kwargs):
        self.split_indices = kwargs.get('split_indices', None)
        self.processed_save_dir = kwargs.get('processed_save_dir', 'processed')
        self.mode = kwargs.get('mode', 'training')
        self.logger_type = kwargs.get('logger_type', 'wandb')
        self.do_early_stopping = kwargs.get('do_early_stopping', True)
        self.params_to_load = kwargs.get('params_to_load', None)
        self.neptune_proj_name = kwargs.get('neptune_proj_name', None)
        self.expt_tags = kwargs.get('expt_tags', [])
        self.log_save_dir = kwargs.get('log_save_dir', 'tensorboard')


def median_absolute_dev(data, median=None):
    """Determine the median absolute deviation from the median (MAD)
    of the supplied dataset.

    Args
    ----
    data (array-like): The data to calculate the MAD of.
    median (float, optional): The median value used to calculate the MAD.
        If set to None, the median will be calculated from the supplied data.

    Returns
    -------
    mad (float): The calculated MAD.
    devs (NumPy array): The absolute deviations from the median used to
        calculate the MAD.
    """
    if median is None:
        median = np.median(data)
    devs = np.abs(data - median)
    mad = np.median(devs)
    return (mad, devs)


def _init_dynamics_mats(dim, n_mats, rand_seed, method):
    rng = np.random.default_rng(rand_seed)
    if method == 'special_ortho':
        dyn_mats = special_ortho_group.rvs(dim=dim, size=n_mats, random_state=rand_seed)
    elif method == 'custom_rotation':
        all_mats = []
        for n in range(n_mats):
            block_R = np.zeros((dim, dim))
            theta = rng.random() * np.pi / 2
            cos, sine = (np.cos(theta), np.sin(theta))
            block_R[:2, :2] = np.array([[cos, -sine], [sine, cos]])
            rand_mat = rng.normal(0, 1, (dim, dim))
            Q, _ = np.linalg.qr(rand_mat)
            all_mats.append(Q @ block_R @ Q.T)
        dyn_mats = np.stack(all_mats, axis=0)
    return torch.tensor(dyn_mats).type(torch.FloatTensor)


# -----------------------------------------------------------------------------
# Latent-space reduction and general statistical plotting
# -----------------------------------------------------------------------------

def z_umap(z, n_components=3, n_neighbors=100, min_dist=0.1, metric='cosine', random_state=1,
           output_metric='euclidean'):
    if torch.is_tensor(z):
        z_np = z.cpu().detach().numpy()
    else:
        z_np = z
    z_dim = z_np.shape[2]
    T = z_np.shape[0]
    N = z_np.shape[1]
    z_cat = np.reshape(z_np, (T * N, z_dim), order='F')
    z_umap_obj = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, output_metric=output_metric,
                           min_dist=min_dist, metric=metric, random_state=random_state, low_memory=False, verbose=True)
    z_transformed = z_umap_obj.fit_transform(z_cat)
    z_reduced = np.reshape(z_transformed, (T, N, n_components), order='F')
    return (z_reduced, z_umap_obj)


def save_figure(fig, save_dir, fn, save_svg=True, save_png=True, save_html=False):
    if save_svg:
        svg_path = os.path.join(save_dir, f'{fn}.svg')
        fig.savefig(svg_path, transparent=True, bbox_inches='tight')
    if save_png:
        png_path = os.path.join(save_dir, f'{fn}.png')
        fig.savefig(png_path, bbox_inches='tight')
    if save_html:
        html_path = os.path.join(save_dir, f'{fn}.html')
        pfig = mpl3d_fig_to_plotly_fig(fig, title='test')
        pfig.update_traces(selector=dict(type='scatter3d'), marker=dict(size=8), line=dict(width=5))
        pfig.update_layout(scene=dict(aspectmode='cube'))
        pio.write_html(pfig, html_path, include_plotlyjs=True, full_html=True, config={'responsive': True})


def plot_scatter(group_stats, params, ax, line_ext, rng, n_boot=1000, alpha=0.05, plot_stats=True, plot_unity=False,
                 plot_boot_band=True, n_boot_band=10000, ci=95, seed=0):
    metric = params['metric']
    u_key = f'u_{metric}'
    m_key = f'm_{metric}'
    u_vals = np.array(group_stats[u_key])
    m_vals = np.array(group_stats[m_key])
    plot_x = np.array([min(u_vals) - line_ext, max(u_vals) + line_ext])
    m, b = np.polyfit(u_vals, m_vals, 1)
    ax.plot(plot_x, m * plot_x + b, zorder=1, linewidth=0.5, color='#E64B35')
    if plot_unity:
        ax.plot(plot_x, plot_x, zorder=1, linewidth=0.5, color='black', linestyle='--')
    ax.scatter(u_vals, m_vals, s=4.0, marker='o', zorder=2, alpha=0.8, color='#377EB8')
    ax.set_xlabel(f"Participant {params['label']}")
    ax.set_ylabel(f"Model {params['label']}")
    r, p, ci_lo, ci_hi = pearson_bootstrap(u_vals, m_vals, rng, n_boot=n_boot, alpha=alpha)
    u_mean = np.mean(u_vals)
    m_mean = np.mean(m_vals)
    u_sem = np.std(u_vals) / np.sqrt(len(u_vals))
    m_sem = np.std(m_vals) / np.sqrt(len(m_vals))
    if 'switch_cost_trial_type_2' in metric:
        metric = 'E2G switch cost'
    elif 'switch_cost_trial_type_3' in metric:
        metric = 'G2E switch cost'
    print(f'{metric} stats:')
    p_str = '{:0.2e}'.format(p)
    r_str = f'r = {round(r, 2)}, 95% CI: ({round(ci_lo, 2)}, {round(ci_hi, 2)})'
    print(f'{r_str}, p-value: {p_str}')
    print(f'Best-fit slope: {m}; intercept: {b}')
    print(f'Participant {metric} mean +/- s.e.m.: {u_mean} +/- {u_sem}')
    print(f'Model {metric} mean +/- s.e.m.: {m_mean} +/- {m_sem}')
    print('--------------------------------------------------------')
    if plot_stats:
        ax.text(0.05, 0.95, r_str, transform=ax.transAxes, fontsize=5, verticalalignment='top')
        stars = p_to_stars(p)
        ax.text(0.85, 0.05, stars, transform=ax.transAxes, fontsize=5, verticalalignment='top')
    if plot_boot_band:
        x_pad = 0.05 * (np.max(u_vals) - np.min(u_vals) + 1e-12)
        x_grid = np.linspace(np.min(u_vals) - x_pad, np.max(u_vals) + x_pad, 200)
        y_hat, y_low, y_high = bootstrap_regression_band(u_vals, m_vals, x_grid, n_boot=n_boot_band, ci=ci, seed=seed)
        ax.plot(x_grid, y_hat, linewidth=1, color='r')
        ax.fill_between(x_grid, y_low, y_high, alpha=0.2, color='grey')


def get_stimulus_combos():
    combos = list(itertools.product([0, 1], [0, 1], [0, 1]))
    return combos


def pearson_bootstrap(x, y, rng, n_boot=1000, alpha=0.05):
    x, y = (np.array(x), np.array(y))
    r_true, p = pearsonr(x, y)
    r_boot = np.zeros(n_boot)
    for n in range(n_boot):
        inds = rng.choice(np.size(x), np.size(x))
        x_boot, y_boot = (x[inds], y[inds])
        this_r, _ = pearsonr(x_boot, y_boot)
        r_boot[n] = this_r
    r_boot = np.sort(r_boot)
    ci_ind = int(n_boot * alpha / 2)
    ci_lo, ci_hi = (r_boot[ci_ind], r_boot[-ci_ind + 1])
    return (r_true, p, ci_lo, ci_hi)


def bootstrap_regression_band(x, y, x_grid, n_boot=10000, ci=95, seed=0):
    """
    Bootstrap CI band for OLS regression line y = a + b*x.
    Resamples subjects with replacement, fits OLS, predicts on x_grid,
    returns (y_hat, y_low, y_high).
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)
    b, a = ols_slope_intercept(x, y)
    y_hat = a + b * x_grid
    preds = np.zeros((int(n_boot), len(x_grid)), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.randint(0, n, size=n)
        b_i, a_i = ols_slope_intercept(x[idx], y[idx])
        preds[i, :] = a_i + b_i * x_grid
    alpha = (100.0 - ci) / 2.0
    y_low = np.percentile(preds, alpha, axis=0)
    y_high = np.percentile(preds, 100.0 - alpha, axis=0)
    return (y_hat, y_low, y_high)


def ols_slope_intercept(x, y):
    """
    Returns slope, intercept for y ~ a + b*x using closed-form OLS.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = np.mean(x)
    ym = np.mean(y)
    denom = np.sum((x - xm) ** 2)
    if denom <= 1e-12:
        return (np.nan, np.nan)
    b = np.sum((x - xm) * (y - ym)) / denom
    a = ym - b * xm
    return (float(b), float(a))


# -----------------------------------------------------------------------------
# Semantic-axis construction
# -----------------------------------------------------------------------------

def _mean_vec(df, mask, cols):
    m = df.loc[mask, list(cols)]
    if len(m) == 0:
        return np.full(len(cols), np.nan)
    return m.to_numpy(dtype=float).mean(axis=0)


def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(v)
    return v / n


def _proj_out(u, v):
    return v - u * float(np.dot(u, v))


def _safe_det(R):
    try:
        return float(np.linalg.det(R))
    except Exception:
        return np.nan


def _orth_err(R):
    return float(np.linalg.norm(R.T @ R - np.eye(R.shape[0])))


def _cos(a, b, eps=1e-12):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def compute_irrelevant_semantic_axes_raw(df, subj_col='patient', fix_cols=('fix_x', 'fix_y', 'fix_z'), task_col='task',
                                         gender_col='gender', emotion_col='emotion', emotion_task='emotion',
                                         gender_task='gender', gender_levels=('male', 'female'),
                                         emotion_levels=('happy', 'threatening')):
    """
    Build raw semantic axes for rose plots using TASK-IRRELEVANT feature axes.

    Returns
    -------
    axes_raw_irrelevant : dict
        sid -> {
            "task": v_task,
            "emotion": v_emotion_irr,
            "gender": v_gender_irr
        }

    Definitions
    -----------
    task axis:
        mean(fix | task=emotion) - mean(fix | task=gender)

    irrelevant emotion axis:
        mean(fix | task=gender, emotion=threatening) -
        mean(fix | task=gender, emotion=happy)

    irrelevant gender axis:
        mean(fix | task=emotion, gender=female) -
        mean(fix | task=emotion, gender=male)
    """
    df = df.copy()
    g_male, g_female = gender_levels
    e_happy, e_threat = emotion_levels
    axes_raw_irrelevant = {}
    for sid in df[subj_col].unique():
        d = df[df[subj_col] == sid]
        mu_fix_emotiontask = _mean_vec(d, d[task_col] == emotion_task, fix_cols)
        mu_fix_gendertask = _mean_vec(d, d[task_col] == gender_task, fix_cols)
        v_task = mu_fix_emotiontask - mu_fix_gendertask
        mu_fix_threat_GT = _mean_vec(d, (d[task_col] == gender_task) & (d[emotion_col] == e_threat), fix_cols)
        mu_fix_happy_GT = _mean_vec(d, (d[task_col] == gender_task) & (d[emotion_col] == e_happy), fix_cols)
        v_emotion_irr = mu_fix_threat_GT - mu_fix_happy_GT
        mu_fix_female_ET = _mean_vec(d, (d[task_col] == emotion_task) & (d[gender_col] == g_female), fix_cols)
        mu_fix_male_ET = _mean_vec(d, (d[task_col] == emotion_task) & (d[gender_col] == g_male), fix_cols)
        v_gender_irr = mu_fix_female_ET - mu_fix_male_ET
        axes_raw_irrelevant[sid] = {'task': v_task, 'emotion': v_emotion_irr, 'gender': v_gender_irr}
    return axes_raw_irrelevant


def compute_aligned_projections_joint(df, subj_col='patient', cols=('x', 'y', 'z'),
                                      fix_cols=('fix_x', 'fix_y', 'fix_z'), task_col='task', gender_col='gender',
                                      emotion_col='emotion', emotion_task='emotion', gender_task='gender',
                                      gender_levels=('male', 'female'), emotion_levels=('happy', 'threatening'),
                                      axis_order=('task', 'emotion', 'gender'), out_cols=('sx', 'sy', 'sz'),
                                      enforce_signs=True, center_per_subject=False):
    """
    Goal 2 implementation (from scratch, using fixed points + onset points):
      - For each subject, define explicit semantic contrast vectors using FIXED points:
          v_task    = mean(fix | task=emotion) - mean(fix | task=gender)           -> + means "emotion task"
          v_emotion = mean(fix | task=emotion, emotion=threatening) - mean(fix | task=emotion, emotion=happy)
          v_gender  = mean(fix | task=gender, gender=female) - mean(fix | task=gender, gender=male)
      - Orthonormalize to build E_local = [e_task, e_emotion, e_gender] (in axis_order)
      - Rotation matrix R_subj maps original coords -> semantic coords:
          X_sem = (X - mu_subj) @ R_subj   if center_per_subject else X @ R_subj
        where R_subj = E_local (if treating columns as semantic axes in original space) OR E_local.T depending on convention.
        Here we use:  semantic coords = X @ E_local
        because columns of E_local are semantic axes in original space.

    Returns:
      aligned_df : df with semantic-aligned coords in out_cols
      E_refs     : dict with global basis definition (here it's identity per axis_order; kept for compatibility)
      diags      : per-subject diagnostics, including gender residual norm
                   and cross-product fallback flags
      subj_Rs    : per-subject rotation matrix (D x D) to apply to ANY points in original space
                  using: X_sem = X @ subj_Rs[sid]   (or (X-mu)@R if centering enabled)
    """
    df = df.copy()
    D = len(cols)
    assert D == 3, 'This implementation assumes 3D UMAP (x,y,z).'
    assert len(fix_cols) == 3, 'fix_cols must be 3D as well.'
    assert len(out_cols) == 3
    base_names = ('task', 'emotion', 'gender')
    name_to_idx = {n: i for i, n in enumerate(base_names)}
    perm = [name_to_idx[n] for n in axis_order]
    subj_ids = df[subj_col].unique()
    subj_Rs = {}
    diags = {}
    outs = []
    g_male, g_female = gender_levels
    e_happy, e_threat = emotion_levels
    axes_raw = {}
    for sid in subj_ids:
        d = df[df[subj_col] == sid]
        mu_fix_emotiontask = _mean_vec(d, d[task_col] == emotion_task, fix_cols)
        mu_fix_gendertask = _mean_vec(d, d[task_col] == gender_task, fix_cols)
        v_task = mu_fix_emotiontask - mu_fix_gendertask
        mu_fix_threat_ET = _mean_vec(d, (d[task_col] == emotion_task) & (d[emotion_col] == e_threat), fix_cols)
        mu_fix_happy_ET = _mean_vec(d, (d[task_col] == emotion_task) & (d[emotion_col] == e_happy), fix_cols)
        v_emotion = mu_fix_threat_ET - mu_fix_happy_ET
        mu_fix_female_GT = _mean_vec(d, (d[task_col] == gender_task) & (d[gender_col] == g_female), fix_cols)
        mu_fix_male_GT = _mean_vec(d, (d[task_col] == gender_task) & (d[gender_col] == g_male), fix_cols)
        v_gender = mu_fix_female_GT - mu_fix_male_GT
        task_strength = float(np.linalg.norm(v_task)) if np.all(np.isfinite(v_task)) else np.nan
        emotion_strength = float(np.linalg.norm(v_emotion)) if np.all(np.isfinite(v_emotion)) else np.nan
        gender_strength = float(np.linalg.norm(v_gender)) if np.all(np.isfinite(v_gender)) else np.nan
        e_task = _unit(v_task)
        e_emotion = _unit(_proj_out(e_task, v_emotion))
        v_gender_ortho = _proj_out(e_task, v_gender)
        v_gender_ortho = _proj_out(e_emotion, v_gender_ortho)
        gender_resid_norm = float(np.linalg.norm(v_gender_ortho))
        gender_degenerate = not np.isfinite(gender_resid_norm) or gender_resid_norm < 1e-12
        e_gender = _unit(v_gender_ortho)
        gender_crossprod_fallback = False
        if gender_degenerate and np.linalg.norm(e_task) > 1e-08 and (np.linalg.norm(e_emotion) > 1e-08):
            e_gender = _unit(np.cross(e_task, e_emotion))
            gender_crossprod_fallback = True
        emotion_crossprod_fallback = False
        if np.linalg.norm(e_emotion) < 1e-08 and np.linalg.norm(e_task) > 1e-08 and (np.linalg.norm(e_gender) > 1e-08):
            e_emotion = _unit(np.cross(e_gender, e_task))
            emotion_crossprod_fallback = True
        E_local = np.column_stack([e_task, e_emotion, e_gender])
        if enforce_signs:
            if np.dot(E_local[:, 0], v_task) < 0:
                E_local[:, 0] *= -1
            if np.dot(E_local[:, 1], v_emotion) < 0:
                E_local[:, 1] *= -1
            if np.dot(E_local[:, 2], v_gender) < 0:
                E_local[:, 2] *= -1
        E_local = E_local[:, perm]
        R = E_local
        detR = _safe_det(R)
        if np.isfinite(detR) and detR < 0:
            R[:, -1] *= -1
            detR = _safe_det(R)
        subj_Rs[sid] = R
        X = d.loc[:, list(cols)].to_numpy(dtype=float)
        mu = X.mean(axis=0) if center_per_subject else np.zeros(3)
        Xc = X - mu
        X_sem = Xc @ R
        tmp = d.copy()
        tmp.loc[:, list(out_cols)] = X_sem
        outs.append(tmp)
        orth_err = _orth_err(R)
        idx_task = axis_order.index('task')
        idx_em = axis_order.index('emotion')
        idx_gen = axis_order.index('gender')
        leak = {'cos_task_to_task': _cos(v_task, R[:, idx_task]), 'cos_task_to_emotion': _cos(v_task, R[:, idx_em]),
                'cos_task_to_gender': _cos(v_task, R[:, idx_gen]),
                'cos_emotion_to_task': _cos(v_emotion, R[:, idx_task]),
                'cos_emotion_to_emotion': _cos(v_emotion, R[:, idx_em]),
                'cos_emotion_to_gender': _cos(v_emotion, R[:, idx_gen]),
                'cos_gender_to_task': _cos(v_gender, R[:, idx_task]),
                'cos_gender_to_emotion': _cos(v_gender, R[:, idx_em]),
                'cos_gender_to_gender': _cos(v_gender, R[:, idx_gen])}
        diags[sid] = {'task_strength': task_strength, 'emotion_strength': emotion_strength,
                      'gender_strength': gender_strength, 'gender_resid_norm': gender_resid_norm,
                      'gender_degenerate': bool(gender_degenerate),
                      'gender_crossprod_fallback': bool(gender_crossprod_fallback),
                      'emotion_crossprod_fallback': bool(emotion_crossprod_fallback), 'orth_err': orth_err, 'det': detR,
                      'centered': bool(center_per_subject), **leak}
        axes_raw[sid] = {'task': v_task, 'emotion': v_emotion, 'gender': v_gender}
    aligned_df = pd.concat(outs, ignore_index=True)
    E_refs = {'axis_order': axis_order, 'E_ref': np.eye(3)}
    return (aligned_df, E_refs, diags, subj_Rs, axes_raw)


# -----------------------------------------------------------------------------
# Trajectory-alignment analysis
# -----------------------------------------------------------------------------

def velocity_from_positions(t, pos):
    """
    Numerical derivative using np.gradient (works for non-uniform t).
    t: (T,) ; pos: (T, D) -> vel: (T, D)
    """
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    D = pos.shape[1]
    vel_cols = []
    for d in range(D):
        vel_cols.append(np.gradient(pos[:, d], t))
    return np.vstack(vel_cols).T


def trial_averaged_trajectory(exp, patient_R, task_cue, prev_task_cue, stim_gender=None, stim_emotion=None,
                              coord_cols=('x', 'y', 'z'), t_col='t', t_start=Constants.T_START, t_end=Constants.T_RT,
                              ndt_df=None):
    """
    Simplified: assumes all selected UMAP trajectories already share the same time length/grid.

    Uses:
      latent_traj = exp.windowed["umap_latents"][:, idx, :]  # (T, Nsel, D_umap)

    Returns
    -------
    t_grid : np.ndarray, shape (T,)  (or (grid_n,) if you have exp.windowed[t_col])
    mean_pos : np.ndarray, shape (T, D)
    idx : pd.Index
    """
    df = exp.df
    mask = (df['task_cue'] == task_cue) & (df['prev_task_cue'] == prev_task_cue)
    if stim_gender is not None:
        mask = mask & (df['stim_gender'] == stim_gender)
    if stim_emotion is not None:
        mask = mask & (df['stim_emotion'] == stim_emotion)
    idx = df.index[mask]
    if idx.empty:
        raise ValueError(
            'No data for condition: task_cue={}, prev_task_cue={}, stim_gender={}, stim_emotion={}'.format(task_cue,
                                                                                                           prev_task_cue,
                                                                                                           stim_gender,
                                                                                                           stim_emotion))
    if ndt_df is not None:
        if task_cue == 0 and task_cue == prev_task_cue:
            ndt = ndt_df.loc[ndt_df['trial'] == 'Gender Repeat Trials', 'mean'].iloc[0]
        elif task_cue == 1 and task_cue == prev_task_cue:
            ndt = ndt_df.loc[ndt_df['trial'] == 'Emotion Repeat Trials', 'mean'].iloc[0]
        elif task_cue == 0 and prev_task_cue == 1:
            ndt = ndt_df.loc[ndt_df['trial'] == 'Gender Switch Trials', 'mean'].iloc[0]
        elif task_cue == 1 and prev_task_cue == 0:
            ndt = ndt_df.loc[ndt_df['trial'] == 'Emotion Switch Trials', 'mean'].iloc[0]
        else:
            raise ValueError('there is no case of trials')
        idx_ndt = np.round(ndt * 1000 / exp.step).astype(int)
    projected_latents = exp.windowed['latents']
    selected_latents = projected_latents[:, idx, :]
    m_rts = df['mrt_ms'].to_numpy()
    rts = np.round(np.mean(m_rts[idx]) / exp.step).astype('int')
    ind_rt = rts + exp.n_pre
    latent_traj = selected_latents
    mean_pos = np.mean(latent_traj, axis=1)
    idx_start = exp.n_pre
    if t_start == Constants.T_HALF_RT:
        idx_start = ind_rt // 2
    elif t_start == Constants.T_NDT:
        idx_start = idx_ndt + exp.n_pre
    idx_end = ind_rt
    if t_end == Constants.T_HALF_RT:
        idx_end = ind_rt // 2
    elif t_end == Constants.T_NDT:
        idx_end = idx_ndt + exp.n_pre
    mean_pos = mean_pos[idx_start:idx_end]
    if isinstance(exp.windowed, dict) and t_col in exp.windowed:
        t_grid = np.asarray(exp.windowed[t_col], dtype=float)
    else:
        t_grid = np.arange(mean_pos.shape[0], dtype=float)
    return (t_grid, mean_pos, idx)


def cosine_timecourse(v1, v2, eps=1e-12, return_weights=False, w_mode='sum', return_cbar_w=False):
    """
    Compute cosine similarity timecourse between two velocity trajectories.

    Parameters
    ----------
    v1, v2 : array-like, shape (T, D)
        Two trajectories on the same time grid.
    eps : float
        Numerical stability for norms.
    return_weights : bool
        If True, also return per-time speed weights w(t).
    w_mode : {"sum","prod","min","mean"}
        How to compute speed weights from speeds s1(t)=||v1(t)|| and s2(t)=||v2(t)||.
    return_cbar_w : bool
        If True, also return the speed-weighted mean cosine c̄_w.

    Returns
    -------
    cos_t : (T,)
        Cosine similarity at each time point.
    w_t : (T,), optional
        Speed weights at each time point (if return_weights=True).
    cbar_w : float, optional
        Speed-weighted mean cosine (if return_cbar_w=True).
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    n1_safe = np.maximum(n1, eps)
    n2_safe = np.maximum(n2, eps)
    u1 = v1 / n1_safe[:, None]
    u2 = v2 / n2_safe[:, None]
    cos_t = np.sum(u1 * u2, axis=1)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    if not (return_weights or return_cbar_w):
        return cos_t
    if w_mode == 'sum':
        w_t = n1 + n2
    elif w_mode == 'prod':
        w_t = n1 * n2
    elif w_mode == 'min':
        w_t = np.minimum(n1, n2)
    elif w_mode == 'mean':
        w_t = 0.5 * (n1 + n2)
    else:
        raise ValueError(f'Unknown w_mode={w_mode}')
    w_t = np.asarray(w_t, dtype=float)
    w_t[~np.isfinite(w_t)] = 0.0
    w_t = np.maximum(w_t, 0.0)
    if not return_cbar_w:
        return (cos_t, w_t)
    m = np.isfinite(cos_t) & np.isfinite(w_t) & (w_t > 0)
    if np.any(m):
        cbar_w = float(np.sum(w_t[m] * cos_t[m]) / np.sum(w_t[m]))
    else:
        cbar_w = np.nan
    return (cos_t, w_t, cbar_w)


def condition_velocity(exp, patient_R, task_cue, prev_task_cue, stim_gender=None, stim_emotion=None,
                       coord_cols=('x', 'y', 'z'), t_col='t', t_start='START', t_end='RT', ndt_df=None):
    """
    Uses self._trial_averaged_trajectory(...) to get mean_pos, then returns velocity.
    Returns: t (T,), vel (T, D), idx (pd.Index)
    """
    t, mean_pos, idx = trial_averaged_trajectory(exp=exp, patient_R=patient_R, task_cue=task_cue,
                                                 prev_task_cue=prev_task_cue, stim_gender=stim_gender,
                                                 stim_emotion=stim_emotion, coord_cols=coord_cols, t_col=t_col,
                                                 t_start=t_start, t_end=t_end, ndt_df=ndt_df)
    vel = velocity_from_positions(t, mean_pos)
    return (t, vel, idx)


def pairwise_velocity_cosine_timecourse(exp, patient_R, cond1, cond2, coord_cols=('x', 'y', 'z'), t_col='t', grid_n=100,
                                        eps=1e-12, t_start=Constants.T_START, t_end=Constants.T_RT, ndt_df=None,
                                        smooth_v=False, sg_window=7, sg_polyorder=3, sg_mode='interp'):
    """
    Phase-warped cosine similarity between two condition velocity trajectories,
    using:
      - Savitzky–Golay smoothing on v (optional, mild)
      - PCHIP interpolation in phase space

    Returns
    -------
    phi_shared : (grid_n,) array in [0, 1]
    cos_phi    : (grid_n,) cosine similarity across phase

    Notes
    -----
    - If time grids are identical, returns absolute-time cosine without phase warp.
    - Savitzky–Golay is applied per-dimension on (T, D) velocity arrays.
    """

    def _sgolay_smooth(v, window, polyorder, mode):
        """
        Mildly smooth velocity along time axis (axis=0).
        Handles short sequences by adapting window length safely.
        """
        v = np.asarray(v, dtype=float)
        T = v.shape[0]
        if T < 3:
            return v
        w = int(window)
        if w < 3:
            return v
        if w > T:
            w = T
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return v
        p = int(polyorder)
        p = min(p, w - 1)
        if p < 1:
            return v
        from scipy.signal import savgol_filter
        return savgol_filter(v, window_length=w, polyorder=p, axis=0, mode=mode)

    t1, v1, _ = condition_velocity(exp, patient_R, cond1['task_cue'], cond1['prev_task_cue'],
                                   stim_gender=cond1.get('stim_gender', None),
                                   stim_emotion=cond1.get('stim_emotion', None), coord_cols=coord_cols, t_col=t_col,
                                   t_start=t_start, t_end=t_end, ndt_df=ndt_df)
    t2, v2, _ = condition_velocity(exp, patient_R, cond2['task_cue'], cond2['prev_task_cue'],
                                   stim_gender=cond2.get('stim_gender', None),
                                   stim_emotion=cond2.get('stim_emotion', None), coord_cols=coord_cols, t_col=t_col,
                                   t_start=t_start, t_end=t_end, ndt_df=ndt_df)
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if smooth_v:
        v1 = _sgolay_smooth(v1, sg_window, sg_polyorder, sg_mode)
        v2 = _sgolay_smooth(v2, sg_window, sg_polyorder, sg_mode)
    if len(t1) == len(t2) and np.allclose(t1, t2):
        cos_t = cosine_timecourse(v1, v2, eps=eps)
        return (t1, cos_t)
    t1_start, t1_end = (float(np.min(t1)), float(np.max(t1)))
    t2_start, t2_end = (float(np.min(t2)), float(np.max(t2)))
    t0 = max(t1_start, t2_start)
    denom1 = t1_end - t0
    denom2 = t2_end - t0
    if denom1 <= 0 or denom2 <= 0:
        raise ValueError('Invalid inferred window: need end > start for both conditions.')
    phi1 = (t1 - t0) / denom1
    phi2 = (t2 - t0) / denom2
    m1 = (phi1 >= 0.0) & (phi1 <= 1.0)
    m2 = (phi2 >= 0.0) & (phi2 <= 1.0)
    phi1, v1 = (phi1[m1], v1[m1])
    phi2, v2 = (phi2[m2], v2[m2])
    if len(phi1) < 2 or len(phi2) < 2:
        raise ValueError('Not enough points after phase mapping to interpolate.')
    phi_shared = np.linspace(0.0, 1.0, int(grid_n))

    def _interp_pchip(phi_src, v_src, phi_new):
        from scipy.interpolate import PchipInterpolator
        order = np.argsort(phi_src)
        x = np.asarray(phi_src[order], dtype=float)
        Y = np.asarray(v_src[order], dtype=float)
        x_unique, idx = np.unique(x, return_index=True)
        Y_unique = Y[idx]
        if len(x_unique) < 2:
            raise ValueError('Need at least 2 unique phase points for PCHIP interpolation.')
        out = np.empty((len(phi_new), Y_unique.shape[1]), dtype=float)
        for d in range(Y_unique.shape[1]):
            f = PchipInterpolator(x_unique, Y_unique[:, d], extrapolate=False)
            out[:, d] = f(phi_new)
            nans = np.isnan(out[:, d])
            if np.any(nans):
                phi_clip = np.clip(phi_new, x_unique[0], x_unique[-1])
                f2 = PchipInterpolator(x_unique, Y_unique[:, d], extrapolate=True)
                out[nans, d] = f2(phi_clip[nans])
        return out

    v1s = _interp_pchip(phi1, v1, phi_shared)
    v2s = _interp_pchip(phi2, v2, phi_shared)
    cos_phi = cosine_timecourse(v1s, v2s, eps=eps)
    return (phi_shared, cos_phi)


def mean_cos(cos_tc):
    """Mean cosine similarity across time, with safe clipping."""
    c = np.asarray(cos_tc, dtype=float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return np.nan
    c = np.clip(c, -1.0, 1.0)
    return float(np.mean(c))


def compute_subject_theta_switch(exp, patient_R=None, coord_cols=('x', 'y', 'z'), t_col='t', grid_n=100,
                                 t_start=Constants.T_START, t_end=Constants.T_RT, ndt_df=None):
    """
    Per subject:
      Theta_g2e = 0.5 * ( mean_angle(female-angry (4b) vs female-happy (3b)) + mean_angle(male-angry (2b) vs male-happy (1b)) )
      Theta_e2g = 0.5 * ( mean_angle(female-happy (2a) vs male-happy (1a)) + mean_angle(female-angry (4a) vs male-angry (3a)) )
      DeltaTheta = Theta_g2e - Theta_e2g

    Uses YOUR condition definitions:
      e2g (a):
        1a male/happy, 2a female/happy, 3a male/angry, 4a female/angry
      g2e (b):
        1b male/happy, 2b male/angry,  3b female/happy, 4b female/angry

    Note: This plan uses UNSIGNED angles only (simple, not heavy circular stats).
    """
    G2E = {'task_cue': 1, 'prev_task_cue': 0}
    E2G = {'task_cue': 0, 'prev_task_cue': 1}
    HAPPY = 0
    ANGRY = 1
    MALE = 0
    FEMALE = 1
    c_m_h_e2g = dict(E2G, stim_gender=MALE, stim_emotion=HAPPY)
    c_f_h_e2g = dict(E2G, stim_gender=FEMALE, stim_emotion=HAPPY)
    c_m_a_e2g = dict(E2G, stim_gender=MALE, stim_emotion=ANGRY)
    c_f_a_e2g = dict(E2G, stim_gender=FEMALE, stim_emotion=ANGRY)
    c_h_e2g = dict(E2G, stim_emotion=HAPPY)
    c_a_e2g = dict(E2G, stim_emotion=ANGRY)
    c_m_e2g = dict(E2G, stim_gender=MALE)
    c_f_e2g = dict(E2G, stim_gender=FEMALE)
    c_m_h_g2e = dict(G2E, stim_gender=MALE, stim_emotion=HAPPY)
    c_m_a_g2e = dict(G2E, stim_gender=MALE, stim_emotion=ANGRY)
    c_f_h_g2e = dict(G2E, stim_gender=FEMALE, stim_emotion=HAPPY)
    c_f_a_g2e = dict(G2E, stim_gender=FEMALE, stim_emotion=ANGRY)
    c_m_g2e = dict(G2E, stim_gender=MALE)
    c_f_g2e = dict(G2E, stim_gender=FEMALE)
    c_h_g2e = dict(G2E, stim_emotion=HAPPY)
    c_a_g2e = dict(G2E, stim_emotion=ANGRY)
    _, cos_f_e_g2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_a_g2e, c_f_h_g2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_f_e_g2e = mean_cos(cos_f_e_g2e)
    _, cos_m_e_g2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_a_g2e, c_m_h_g2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_m_e_g2e = mean_cos(cos_m_e_g2e)
    theta_g2e = 0.5 * (theta_f_e_g2e + theta_m_e_g2e)
    _, cos_g_h_g2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_h_g2e, c_f_h_g2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_h_g2e = mean_cos(cos_g_h_g2e)
    _, cos_g_a_g2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_a_g2e, c_f_a_g2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_a_g2e = mean_cos(cos_g_a_g2e)
    _, cos_e_g2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_h_g2e, c_a_g2e, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_e_g2e = mean_cos(cos_e_g2e)
    _, cos_g_g2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_g2e, c_f_g2e, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_g_g2e = mean_cos(cos_g_g2e)
    _, cos_g_a_e2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_a_e2g, c_m_a_e2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_a_e2g = mean_cos(cos_g_a_e2g)
    _, cos_g_h_e2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_h_e2g, c_m_h_e2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_h_e2g = mean_cos(cos_g_h_e2g)
    theta_e2g = 0.5 * (theta_g_a_e2g + theta_g_h_e2g)
    _, cos_m_e_e2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_h_e2g, c_m_a_e2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_m_e_e2g = mean_cos(cos_m_e_e2g)
    _, cos_f_e_e2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_h_e2g, c_f_a_e2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_f_e_e2g = mean_cos(cos_f_e_e2g)
    _, cos_g_e2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_e2g, c_f_e2g, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_g_e2g = mean_cos(cos_g_e2g)
    _, cos_e_e2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_h_e2g, c_a_e2g, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_e_e2g = mean_cos(cos_e_e2g)
    return {'theta_g2e': float(theta_g2e), 'theta_e2g': float(theta_e2g), 'theta_f_e_g2e': float(theta_f_e_g2e),
            'theta_m_e_g2e': float(theta_m_e_g2e), 'theta_g_h_g2e': float(theta_g_h_g2e),
            'theta_g_a_g2e': float(theta_g_a_g2e), 'theta_e_g2e': float(theta_e_g2e), 'theta_g_g2e': float(theta_g_g2e),
            'theta_g_a_e2g': float(theta_g_a_e2g), 'theta_g_h_e2g': float(theta_g_h_e2g),
            'theta_m_e_e2g': float(theta_m_e_e2g), 'theta_f_e_e2g': float(theta_f_e_e2g),
            'theta_g_e2g': float(theta_g_e2g), 'theta_e_e2g': float(theta_e_e2g)}


def compute_subject_theta_repeat(exp, patient_R=None, coord_cols=('x', 'y', 'z'), t_col='t', grid_n=100,
                                 t_start=Constants.T_START, t_end=Constants.T_RT, ndt_df=None):
    """
    Per subject (REPEAT trials):

      Theta_e2e = 0.5 * (
          mean_angle(female-angry vs female-happy) +
          mean_angle(male-angry   vs male-happy)
      )
        where all trials are emotion-repeat (task=emotion, prev=emotion)

      Theta_g2g = 0.5 * (
          mean_angle(female-angry vs male-angry) +
          mean_angle(female-happy vs male-happy)
      )
        where all trials are gender-repeat (task=gender, prev=gender)

    Uses the same velocity-cosine -> unsigned-angle pipeline as compute_subject_theta_asym:
      - self._pairwise_velocity_cosine_timecourse(...)
      - self._mean_unsigned_angle_from_cos(...)

    Returns
    -------
    dict with theta_e2e, theta_g2g, and component thetas for debugging/reporting.
    """
    G2G = {'task_cue': 0, 'prev_task_cue': 0}
    E2E = {'task_cue': 1, 'prev_task_cue': 1}
    HAPPY = 0
    ANGRY = 1
    MALE = 0
    FEMALE = 1
    c_m_h_e2e = dict(E2E, stim_gender=MALE, stim_emotion=HAPPY)
    c_m_a_e2e = dict(E2E, stim_gender=MALE, stim_emotion=ANGRY)
    c_f_h_e2e = dict(E2E, stim_gender=FEMALE, stim_emotion=HAPPY)
    c_f_a_e2e = dict(E2E, stim_gender=FEMALE, stim_emotion=ANGRY)
    c_m_e2e = dict(E2E, stim_gender=MALE)
    c_f_e2e = dict(E2E, stim_gender=FEMALE)
    c_h_e2e = dict(E2E, stim_emotion=HAPPY)
    c_a_e2e = dict(E2E, stim_emotion=ANGRY)
    c_m_h_g2g = dict(G2G, stim_gender=MALE, stim_emotion=HAPPY)
    c_f_h_g2g = dict(G2G, stim_gender=FEMALE, stim_emotion=HAPPY)
    c_m_a_g2g = dict(G2G, stim_gender=MALE, stim_emotion=ANGRY)
    c_f_a_g2g = dict(G2G, stim_gender=FEMALE, stim_emotion=ANGRY)
    c_h_g2g = dict(G2G, stim_emotion=HAPPY)
    c_a_g2g = dict(G2G, stim_emotion=ANGRY)
    c_m_g2g = dict(G2G, stim_gender=MALE)
    c_f_g2g = dict(G2G, stim_gender=FEMALE)
    _, cos_f_e_e2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_a_e2e, c_f_h_e2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_f_e_e2e = mean_cos(cos_f_e_e2e)
    _, cos_m_e_e2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_a_e2e, c_m_h_e2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_m_e_e2e = mean_cos(cos_m_e_e2e)
    theta_e2e = 0.5 * (theta_f_e_e2e + theta_m_e_e2e)
    _, cos_g_h_e2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_h_e2e, c_f_h_e2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_h_e2e = mean_cos(cos_g_h_e2e)
    _, cos_g_a_e2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_a_e2e, c_f_a_e2e, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_a_e2e = mean_cos(cos_g_a_e2e)
    _, cos_g_e2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_e2e, c_f_e2e, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_g_e2e = mean_cos(cos_g_e2e)
    _, cos_e_e2e = pairwise_velocity_cosine_timecourse(exp, patient_R, c_h_e2e, c_a_e2e, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_e_e2e = mean_cos(cos_e_e2e)
    _, cos_g_a_g2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_a_g2g, c_m_a_g2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_a_g2g = mean_cos(cos_g_a_g2g)
    _, cos_g_h_g2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_h_g2g, c_m_h_g2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_g_h_g2g = mean_cos(cos_g_h_g2g)
    theta_g2g = 0.5 * (theta_g_a_g2g + theta_g_h_g2g)
    _, cos_m_e_g2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_h_g2g, c_m_a_g2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_m_e_g2g = mean_cos(cos_m_e_g2g)
    _, cos_f_e_g2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_f_h_g2g, c_f_a_g2g, coord_cols=coord_cols,
                                                         t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                         ndt_df=ndt_df)
    theta_f_e_g2g = mean_cos(cos_f_e_g2g)
    _, cos_e_g2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_h_g2g, c_a_g2g, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_e_g2g = mean_cos(cos_e_g2g)
    _, cos_g_g2g = pairwise_velocity_cosine_timecourse(exp, patient_R, c_m_g2g, c_f_g2g, coord_cols=coord_cols,
                                                       t_col=t_col, grid_n=grid_n, t_start=t_start, t_end=t_end,
                                                       ndt_df=ndt_df)
    theta_g_g2g = mean_cos(cos_g_g2g)
    return {'theta_e2e': float(theta_e2e), 'theta_g2g': float(theta_g2g), 'theta_f_e_e2e': float(theta_f_e_e2e),
            'theta_m_e_e2e': float(theta_m_e_e2e), 'theta_g_h_e2e': float(theta_g_h_e2e),
            'theta_g_a_e2e': float(theta_g_a_e2e), 'theta_g_e2e': float(theta_g_e2e), 'theta_e_e2e': float(theta_e_e2e),
            'theta_g_a_g2g': float(theta_g_a_g2g), 'theta_g_h_g2g': float(theta_g_h_g2g),
            'theta_m_e_g2g': float(theta_m_e_g2g), 'theta_f_e_g2g': float(theta_f_e_g2g),
            'theta_e_g2g': float(theta_e_g2g), 'theta_g_g2g': float(theta_g_g2g)}


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def spline_smooth(traj, s=0.2):
    from scipy.interpolate import splprep, splev
    tck, u = splprep(traj.T, s=s)
    u_new = np.linspace(0, 1, len(traj))
    x_new, y_new, z_new = splev(u_new, tck)
    return np.vstack((x_new, y_new, z_new)).T


def overlay_semantic_axes_corner(ax, R, corner=(0.08, 0.88, 0.12), scale=0.18, text_offset=1.08, alpha=0.9):
    """
    Draw semantic axes arrows in the corner of a 3D axis,
    with labels at the arrow tips.
    """
    colors = {'Task': Constants.COLOR_TASK_ARROW, 'Emotion': Constants.COLOR_E2E, 'Gender': Constants.COLOR_G2G}
    labels = ['Task', 'Emotion', 'Gender']
    xlim, ylim, zlim = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
    ox = xlim[0] + corner[0] * (xlim[1] - xlim[0])
    oy = ylim[0] + corner[1] * (ylim[1] - ylim[0])
    oz = zlim[0] + corner[2] * (zlim[1] - zlim[0])
    origin = np.array([ox, oy, oz])
    L = scale * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    for i, lab in enumerate(labels):
        v = R[:, i]
        v = v / (np.linalg.norm(v) + 1e-12)
        ax.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2], length=L, color=colors[lab], linewidth=3,
                  alpha=alpha, normalize=True, arrow_length_ratio=0.15)
        tip = origin + v * L
        ax.text(tip[0] * text_offset, tip[1] * text_offset, tip[2] * text_offset, lab, color=colors[lab], fontsize=6,
                weight='bold', ha='center', va='center')
