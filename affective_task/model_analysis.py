"""Core latent-state analyses used by GeoDynFormer manuscript figures."""

import itertools
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

from .utils import get_stimulus_combos


class FixedPointFinder:
    """Find and summarize stable fixed points from a trained model."""

    def __init__(
            self,
            expt,
            expt_stats,
            fp_path,
            fp_summary_path,
            load_saved=True,
            z_dim=16,
            rand_seed=12345,
            t_step=20,
            t_search_win=10000,
            max_sd_tol=1e-3,
            max_dist_tol=1e-3,
            device="cpu",
    ):
        self.expt = expt
        self.expt_stats = expt_stats
        self.fp_path = fp_path
        self.fp_summary_path = fp_summary_path
        self.load_saved = load_saved
        self.z_dim = z_dim
        self.t_step = t_step
        self.t_win = t_search_win
        self.max_sd_tol = max_sd_tol
        self.max_dist_tol = max_dist_tol
        self.stimulus_combos = get_stimulus_combos()
        self.rng = np.random.default_rng(rand_seed)
        self.device = device

    def find_fixed_points(self, N, T):
        """Find stable fixed points from ``N`` initial states over ``T`` ms."""

        if os.path.exists(self.fp_path) and self.load_saved:
            with open(self.fp_path, "rb") as path:
                fps = pickle.load(path)
        else:
            all_fps = []
            # Randomly sample initial states to find fixed points
            z0 = self._make_z0(N).to(self.device)
            for c in self.stimulus_combos:
                stim_gender, stim_emotion, cue = c[0], c[1], c[2]
                c_stimuli = self._make_fp_stimuli(
                    N,
                    T,
                    stim_emotion=stim_emotion,
                    stim_gender=stim_gender,
                    cue=cue,
                ).to(self.device)
                # Generate responses
                _, _, _, z_out, _, _ = self.expt.model.forward(
                    c_stimuli,
                    generate_mode=True,
                    clamp=True,
                    z0_supplied=z0,
                )
                z_np = z_out.cpu().detach().numpy()
                c_fps = self._check_for_fps(
                    z_np,
                    N,
                    stim_emotion=stim_emotion,
                    stim_gender=stim_gender,
                    cue=cue,
                )
                if c_fps is not None:
                    all_fps.extend(c_fps)

            if len(all_fps) > 0:
                fp_df = pd.concat(all_fps)
                fps = self._project_fps(fp_df)
                fps.reset_index(drop=True, inplace=True)
            else:
                fps = pd.DataFrame(
                    {
                        "type": [],
                        "zloc": [],
                        "cue": [],
                        "stim_gender": [],
                        "stim_emotion": [],
                    }
                )
            fps.to_pickle(self.fp_path, protocol=4)
        return fps

    def get_fixed_point_summary(self, fps):
        """
        Summarize fixed-point distances using controlled pairwise contrasts
        designed to test the proposed hierarchical organization:

        1) task_matched:
           different task, same gender, same emotion

        2) task_relevant:
           same task, different task-relevant feature,
           task-irrelevant feature matched

           - gender task: different gender, same emotion
           - emotion task: different emotion, same gender

        3) task_irrelevant:
           same task, different task-irrelevant feature,
           task-relevant feature matched

           - gender task: same gender, different emotion
           - emotion task: same emotion, different gender

        Task-specific relevant/irrelevant contrasts are also retained for
        separate analyses of the gender and emotion tasks.
        """
        if os.path.exists(self.fp_summary_path) and self.load_saved:
            with open(self.fp_summary_path, "rb") as path:
                summary = pickle.load(path)
            return summary

        summary = {
            # Three controlled hierarchy levels
            "task_matched": [],
            "task_relevant": [],
            "task_irrelevant": [],

            # Task-specific controlled within-task contrasts
            "gender_relevant": [],
            "gender_irrelevant": [],
            "emotion_relevant": [],
            "emotion_irrelevant": [],

            # Bookkeeping
            "N": 0,
            "f_stimuli_with_fp": 0,
        }

        if len(fps) == 0:
            with open(self.fp_summary_path, "wb") as path:
                pickle.dump(summary, path, protocol=4)
            return summary

        fp_inds = np.arange(len(fps))
        pairs = list(itertools.combinations(fp_inds, 2))

        for i1, i2 in pairs:
            fp1 = fps.iloc[i1, :]
            fp2 = fps.iloc[i2, :]

            z1, z2 = fp1["zloc"], fp2["zloc"]
            this_dist = np.linalg.norm(z1 - z2, ord=2)

            pair_type = self._classify_pair(fp1, fp2)

            if pair_type is None:
                # Exclude uncontrolled pairs, e.g.:
                # - between-task pairs with non-matching target attributes
                # - within-task pairs differing in both relevant and irrelevant attributes
                continue

            summary[pair_type].append(this_dist)

            if pair_type == "task_relevant":
                if fp1["cue"] == 0:
                    summary["gender_relevant"].append(this_dist)
                else:
                    summary["emotion_relevant"].append(this_dist)

            elif pair_type == "task_irrelevant":
                if fp1["cue"] == 0:
                    summary["gender_irrelevant"].append(this_dist)
                else:
                    summary["emotion_irrelevant"].append(this_dist)

        summary["N"] = len(fps)
        summary["f_stimuli_with_fp"] = self._count_fps(fps)

        with open(self.fp_summary_path, "wb") as path:
            pickle.dump(summary, path, protocol=4)

        return summary

    def _count_fps(self, fps):
        # Determine the fraction of the 8 possible stimulus
        # configurations that have a fixed point.
        n = len(
            fps.drop_duplicates(
                subset=["cue", "stim_gender", "stim_emotion"],
                inplace=False,
            )
        )
        return n / 8

    def _classify_pair(self, fp1, fp2):
        """
        Classify a fixed-point pair into one of the controlled hierarchy
        contrasts. Uncontrolled pairs return None.

        Returns
        -------
        "task_matched"
            Different task, with both target attributes matched.

        "task_relevant"
            Same task, different task-relevant feature, with the
            task-irrelevant feature matched.

        "task_irrelevant"
            Same task, different task-irrelevant feature, with the
            task-relevant feature matched.

        None
            Pair does not isolate a single intended hierarchy level.
        """
        gender1 = fp1["stim_gender"]
        emotion1 = fp1["stim_emotion"]
        cue1 = fp1["cue"]

        gender2 = fp2["stim_gender"]
        emotion2 = fp2["stim_emotion"]
        cue2 = fp2["cue"]

        same_task = cue1 == cue2
        same_gender = gender1 == gender2
        same_emotion = emotion1 == emotion2

        # --------------------------------------------------
        # Level 1: task contrast
        # Different task, both target attributes matched
        # --------------------------------------------------
        if not same_task:
            if same_gender and same_emotion:
                return "task_matched"
            return None

        # --------------------------------------------------
        # Levels 2 and 3: within-task controlled contrasts
        # --------------------------------------------------
        if cue1 == 0:  # gender task
            # Task-relevant contrast:
            # different gender, same emotion
            if (not same_gender) and same_emotion:
                return "task_relevant"

            # Task-irrelevant contrast:
            # same gender, different emotion
            if same_gender and (not same_emotion):
                return "task_irrelevant"

        else:  # emotion task
            # Task-relevant contrast:
            # different emotion, same gender
            if same_gender and (not same_emotion):
                return "task_relevant"

            # Task-irrelevant contrast:
            # same emotion, different gender
            if (not same_gender) and same_emotion:
                return "task_irrelevant"

        # Exclude pairs differing in both gender and emotion,
        # as they do not isolate relevant vs. irrelevant effects.
        return None

    def _make_z0(self, N):
        # Select a set of initial states for finding fixed points:
        # randomly sample z from states visited when the model
        # is used to generate responses.
        z = self.expt_stats.latents
        t_rand = self.rng.choice(z.shape[0], N)
        n_rand = self.rng.choice(z.shape[1], N)
        z0 = z[t_rand, n_rand, :]
        z0_torch = torch.Tensor(np.reshape(z0, (1, N, self.z_dim)))
        return z0_torch

    def _make_fp_stimuli(self, N, T, stim_emotion=None, stim_gender=None, cue=None):
        # Make N static stimuli of length T: one stimulus configuration is
        # active for the entire stimulus duration.
        gender_start_ind, emotion_start_ind, cue_start_ind = 4, 6, 8
        T_samples = T // self.t_step
        stimuli = torch.zeros(T_samples, N, 10)
        if stim_emotion is not None:
            stimuli[:, :, emotion_start_ind + stim_emotion] = 1
        if stim_gender is not None:
            stimuli[:, :, gender_start_ind + stim_gender] = 1
        if cue is not None:
            stimuli[:, :, cue_start_ind + cue] = 1
        return stimuli

    def _check_for_fps(self, z, N, stim_emotion=None, stim_gender=None, cue=None):
        # Find and validate fixed points from the latent state
        win_samples = self.t_win // self.t_step  # window to look for fps
        z_win = z[-win_samples:, :, :]
        z_sd = np.std(z_win, axis=0)
        z_mean_sd = np.mean(z_sd, axis=1)
        stable_inds = np.nonzero(z_mean_sd <= self.max_sd_tol)[0]
        z_means = np.mean(z_win, axis=0)

        if len(stable_inds) == 0:
            fps = None
        else:
            fps = []
            stable_z = []
            fp_type = "stable"
            for si in stable_inds:
                si_z = z_means[si, :]
                if len(stable_z) > 0:
                    new_fp = self._check_redundant_fp(stable_z, si_z)
                else:
                    new_fp = True
                if new_fp:
                    stable_z.append(si_z)
                    this_fp = pd.DataFrame(
                        {
                            "type": fp_type,
                            "zloc": [si_z],
                            "cue": cue,
                            "stim_gender": stim_gender,
                            "stim_emotion": stim_emotion,
                        }
                    )
                    fps.append(this_fp)
        return fps

    def _check_redundant_fp(self, all_z, test_z):
        # Check to see if fixed point is redundant with existing fixed points
        new_fp = True
        for z in all_z:
            this_dist = np.linalg.norm(z - test_z, ord=2)
            if this_dist <= self.max_dist_tol:
                new_fp = False
        return new_fp

    def _project_fps(self, fps, keep_dim=3):
        # Project the fixed points onto the latent axes (PC or UMAP axes)
        umap_obj = self.expt_stats.umap_obj
        umap_fps_proj = []
        for fp in fps["zloc"]:
            # UMAP
            this_proj_umap = umap_obj.transform(fp.reshape((1, len(fp))))
            umap_fps_proj.append(this_proj_umap[0][:keep_dim])
        # UMAP
        fps["zloc_umap"] = umap_fps_proj
        return fps


class LatentsLDA:
    """Run the manuscript LDA analyses in the native latent space."""

    def __init__(
            self,
            expt_stats,
            save_path,
            load_saved=True,
            time_range=[-100, 1600],
            n_shuffle=100,
            rand_seed=12345,
            cv_folds=5,
    ):
        self.expt_stats = expt_stats
        self.save_path = save_path
        self.load_saved = load_saved
        mean_rt = self.expt_stats.summary_stats["m_mean_rt"]
        self.t_ind = np.argmin(np.absolute(self.expt_stats.t_axis - mean_rt))
        self.n_shuffle = n_shuffle
        self.rand_seed = rand_seed
        self.cv_folds = cv_folds
        self.rng = np.random.default_rng(rand_seed)

        # Retained for compatibility with the preprocessing API.
        _ = time_range
        self._split_trials_by_task()

    def _split_trials_by_task(self):
        gender_filt = {"task_cue": 0}
        emotion_filt = {"task_cue": 1}
        gender_inds = self.expt_stats.select(**gender_filt)
        emotion_inds = self.expt_stats.select(**emotion_filt)

        self.gender_z = np.asarray(
            self.expt_stats.windowed["latents"][self.t_ind, gender_inds, :]
        )
        self.emotion_z = np.asarray(
            self.expt_stats.windowed["latents"][self.t_ind, emotion_inds, :]
        )

        self.gender_z = np.atleast_2d(self.gender_z)
        self.emotion_z = np.atleast_2d(self.emotion_z)

    def _get_lda_error(self, x, y):
        """
        Cross-validated LDA misclassification rate.

        The original implementation fit and scored the LDA on the same
        observations. Here, each observation is scored only when held out
        from model fitting. Because all binary comparisons are balanced
        before this function is called, chance misclassification is 0.5.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("LDA analysis requires exactly two classes.")

        min_class_n = int(np.min(counts))
        n_splits = min(self.cv_folds, min_class_n)

        if n_splits < 2:
            raise ValueError(
                "At least two observations per class are required "
                "for cross-validated LDA."
            )

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.rand_seed,
        )

        fold_errors = []

        for train_inds, test_inds in cv.split(x, y):
            this_lda = LDA()
            this_lda.fit(x[train_inds], y[train_inds])
            fold_error = 1 - this_lda.score(
                x[test_inds],
                y[test_inds],
            )
            fold_errors.append(fold_error)

        return float(np.mean(fold_errors))

    def run_lda_analysis(self):
        if os.path.exists(self.save_path) and self.load_saved:
            with open(self.save_path, "rb") as path:
                lda_info = pickle.load(path)
            return lda_info

        # --------------------------------------------------
        # Helper functions
        # --------------------------------------------------
        def _get_latents(inds):
            z = np.asarray(
                self.expt_stats.windowed["latents"][self.t_ind, inds, :]
            )
            return np.atleast_2d(z)

        def _balance_binary_data(x1, x2):
            """
            Equalize the two class sizes by random downsampling.

            Balancing is performed once for a given observed comparison.
            The same balanced observations are then used for both the
            observed cross-validated error and all label permutations.
            """
            x1 = np.atleast_2d(np.asarray(x1))
            x2 = np.atleast_2d(np.asarray(x2))

            n = min(x1.shape[0], x2.shape[0])

            if n < 2:
                raise ValueError(
                    "At least two observations are required in each class "
                    "for balanced cross-validated LDA."
                )

            if x1.shape[0] > n:
                inds1 = self.rng.choice(
                    x1.shape[0],
                    size=n,
                    replace=False,
                )
                x1 = x1[inds1, :]

            if x2.shape[0] > n:
                inds2 = self.rng.choice(
                    x2.shape[0],
                    size=n,
                    replace=False,
                )
                x2 = x2[inds2, :]

            return x1, x2

        def _make_xy(x1, x2):
            x = np.concatenate((x1, x2), axis=0)
            y = np.hstack(
                (
                    np.zeros(x1.shape[0], dtype=int),
                    np.ones(x2.shape[0], dtype=int),
                )
            )
            return x, y

        def _get_shuffle_error_dist(x, y):
            """
            Empirical null distribution from label permutation.

            Labels are permuted across the same balanced observations used
            for the observed analysis. Each permuted-label dataset is then
            evaluated with the same cross-validated LDA procedure.
            """
            shuffle_errors = []

            for _ in range(self.n_shuffle):
                y_perm = self.rng.permutation(y)
                shuffle_errors.append(
                    self._get_lda_error(x, y_perm)
                )

            return np.asarray(shuffle_errors)

        def _percentile_against_shuffle(true_error, shuffle_dist):
            """
            Lower LDA error means better classification.

            Returns the percentile of the observed error relative
            to the empirical null distribution.
            """
            percentile = 100 * np.mean(
                shuffle_dist <= true_error
            )

            p_perm = (
                             np.sum(shuffle_dist <= true_error) + 1
                     ) / (
                             len(shuffle_dist) + 1
                     )

            return {
                "shuffle_mean": float(
                    np.mean(shuffle_dist)
                ),
                "shuffle_sem": float(
                    np.std(shuffle_dist, ddof=1)
                    / np.sqrt(len(shuffle_dist))
                ),
                "shuffle_min": float(
                    np.min(shuffle_dist)
                ),
                "shuffle_max": float(
                    np.max(shuffle_dist)
                ),
                "shuffle_p05": float(
                    np.percentile(shuffle_dist, 5)
                ),
                "shuffle_p95": float(
                    np.percentile(shuffle_dist, 95)
                ),
                "observed_percentile": float(
                    percentile
                ),
                "permutation_p_lower": float(
                    p_perm
                ),
                "observed_below_all_shuffles": bool(
                    true_error < np.min(shuffle_dist)
                ),
            }

        def _lda_array_error(x1, x2):
            """
            Balanced, cross-validated binary LDA comparison.
            """
            x1, x2 = _balance_binary_data(x1, x2)
            x, y = _make_xy(x1, x2)

            true_error = self._get_lda_error(x, y)

            shuffle_dist = _get_shuffle_error_dist(
                x,
                y,
            )
            shuffle_error = float(np.mean(shuffle_dist))

            shuffle_stats = _percentile_against_shuffle(
                true_error,
                shuffle_dist,
            )

            return (
                true_error,
                shuffle_error,
                shuffle_dist,
                shuffle_stats,
                int(x1.shape[0]),
            )

        def _lda_pair_error(x1_inds, x2_inds):
            x1 = _get_latents(x1_inds)
            x2 = _get_latents(x2_inds)
            return _lda_array_error(x1, x2)

        def _condition_inds(
                task_cue,
                stim_gender,
                stim_emotion,
        ):
            return np.asarray(
                self.expt_stats.select(
                    task_cue=task_cue,
                    stim_gender=stim_gender,
                    stim_emotion=stim_emotion,
                ),
                dtype=int,
            )

        def _aggregate_pairwise_lda(pair_list):
            """
            Run multiple matched, balanced cross-validated LDA comparisons
            and average them.

            For the empirical null, the pairwise shuffled errors are
            averaged within each permutation iteration, matching the way
            the observed statistic is formed.
            """
            pair_true_errors = []
            pair_shuffle_errors = []
            pair_shuffle_dists = []
            pair_shuffle_stats = []
            pair_class_n = []

            for x1_inds, x2_inds in pair_list:
                if len(x1_inds) < 2 or len(x2_inds) < 2:
                    continue

                (
                    true_error,
                    shuffle_error,
                    shuffle_dist,
                    shuffle_stats,
                    class_n,
                ) = _lda_pair_error(
                    x1_inds,
                    x2_inds,
                )

                pair_true_errors.append(true_error)
                pair_shuffle_errors.append(shuffle_error)
                pair_shuffle_dists.append(shuffle_dist)
                pair_shuffle_stats.append(shuffle_stats)
                pair_class_n.append(class_n)

            if len(pair_true_errors) == 0:
                raise ValueError(
                    "No valid LDA comparisons were available."
                )

            mean_true_error = float(
                np.mean(pair_true_errors)
            )

            aggregate_shuffle_dist = np.mean(
                np.vstack(pair_shuffle_dists),
                axis=0,
            )

            mean_shuffle_error = float(
                np.mean(aggregate_shuffle_dist)
            )

            aggregate_shuffle_stats = (
                _percentile_against_shuffle(
                    mean_true_error,
                    aggregate_shuffle_dist,
                )
            )

            return (
                mean_true_error,
                mean_shuffle_error,
                aggregate_shuffle_dist,
                aggregate_shuffle_stats,
                pair_true_errors,
                pair_shuffle_errors,
                pair_shuffle_dists,
                pair_shuffle_stats,
                pair_class_n,
            )

        # --------------------------------------------------
        # Fig. 5b: between-task discrimination
        # --------------------------------------------------
        # Gender-task vs. emotion-task trials. The four gender x emotion
        # feature combinations are pooled within each task. The two task
        # classes are then equalized before cross-validation.
        (
            bw_error,
            bw_shuffle_error,
            bw_shuffle_dist,
            bw_shuffle_stats,
            bw_class_n,
        ) = _lda_array_error(
            self.gender_z,
            self.emotion_z,
        )

        # --------------------------------------------------
        # Fig. 5c: task-relevant feature discrimination
        # with the task-irrelevant feature controlled
        # --------------------------------------------------

        # Gender task:
        # male vs. female while holding emotion fixed.
        gender_male_happy_inds = _condition_inds(
            task_cue=0,
            stim_gender=0,
            stim_emotion=0,
        )
        gender_female_happy_inds = _condition_inds(
            task_cue=0,
            stim_gender=1,
            stim_emotion=0,
        )

        gender_male_angry_inds = _condition_inds(
            task_cue=0,
            stim_gender=0,
            stim_emotion=1,
        )
        gender_female_angry_inds = _condition_inds(
            task_cue=0,
            stim_gender=1,
            stim_emotion=1,
        )

        gender_pairs = [
            (
                gender_male_happy_inds,
                gender_female_happy_inds,
            ),
            (
                gender_male_angry_inds,
                gender_female_angry_inds,
            ),
        ]

        (
            gender_error,
            shuffle_gender_error,
            gender_shuffle_dist,
            gender_shuffle_stats,
            gender_pair_errors,
            gender_pair_shuffle_errors,
            gender_pair_shuffle_dists,
            gender_pair_shuffle_stats,
            gender_pair_class_n,
        ) = _aggregate_pairwise_lda(
            gender_pairs
        )

        # Emotion task:
        # happy vs. angry while holding gender fixed.
        emotion_male_happy_inds = _condition_inds(
            task_cue=1,
            stim_gender=0,
            stim_emotion=0,
        )
        emotion_male_angry_inds = _condition_inds(
            task_cue=1,
            stim_gender=0,
            stim_emotion=1,
        )

        emotion_female_happy_inds = _condition_inds(
            task_cue=1,
            stim_gender=1,
            stim_emotion=0,
        )
        emotion_female_angry_inds = _condition_inds(
            task_cue=1,
            stim_gender=1,
            stim_emotion=1,
        )

        emotion_pairs = [
            (
                emotion_male_happy_inds,
                emotion_male_angry_inds,
            ),
            (
                emotion_female_happy_inds,
                emotion_female_angry_inds,
            ),
        ]

        (
            emotion_error,
            shuffle_emotion_error,
            emotion_shuffle_dist,
            emotion_shuffle_stats,
            emotion_pair_errors,
            emotion_pair_shuffle_errors,
            emotion_pair_shuffle_dists,
            emotion_pair_shuffle_stats,
            emotion_pair_class_n,
        ) = _aggregate_pairwise_lda(
            emotion_pairs
        )

        # --------------------------------------------------
        # Fig. 5b: within-task discrimination
        # All pairwise condition discriminations within each task
        # --------------------------------------------------
        true_within_errors = []
        shuffle_within_errors = []
        within_shuffle_dists = []
        within_shuffle_stats_list = []
        within_pair_class_n = []

        for task_cue in [0, 1]:
            condition_inds = []

            for stim_gender in [0, 1]:
                for stim_emotion in [0, 1]:
                    inds = _condition_inds(
                        task_cue=task_cue,
                        stim_gender=stim_gender,
                        stim_emotion=stim_emotion,
                    )
                    condition_inds.append(inds)

            # Four conditions -> six unique pairwise comparisons.
            for i in range(len(condition_inds)):
                for j in range(i + 1, len(condition_inds)):
                    x1_inds = condition_inds[i]
                    x2_inds = condition_inds[j]

                    if len(x1_inds) < 2 or len(x2_inds) < 2:
                        continue

                    (
                        true_error,
                        shuffle_error,
                        shuffle_dist,
                        shuffle_stats,
                        class_n,
                    ) = _lda_pair_error(
                        x1_inds,
                        x2_inds,
                    )

                    true_within_errors.append(true_error)
                    shuffle_within_errors.append(shuffle_error)
                    within_shuffle_dists.append(shuffle_dist)
                    within_shuffle_stats_list.append(shuffle_stats)
                    within_pair_class_n.append(class_n)

        if len(true_within_errors) == 0:
            raise ValueError(
                "No valid within-task LDA comparisons were available."
            )

        within_error = float(
            np.mean(true_within_errors)
        )

        # Construct the null distribution exactly as the observed statistic:
        # average across all pairwise within-task comparisons within each
        # shuffle iteration.
        within_shuffle_dist = np.mean(
            np.vstack(within_shuffle_dists),
            axis=0,
        )

        within_shuffle_error = float(
            np.mean(within_shuffle_dist)
        )

        within_shuffle_stats = (
            _percentile_against_shuffle(
                within_error,
                within_shuffle_dist,
            )
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------
        lda_info = {
            # Fig. 5b: between task
            "bw_error": bw_error,
            "bw_shuffle_error": bw_shuffle_error,
            "bw_shuffle_dist": bw_shuffle_dist,
            "bw_shuffle_stats": bw_shuffle_stats,

            # Fig. 5b: within task
            "within_error": within_error,
            "within_shuffle_error": within_shuffle_error,
            "within_shuffle_dist": within_shuffle_dist,
            "within_shuffle_stats": within_shuffle_stats,

            # Fig. 5c: controlled gender classification
            "gender_error": gender_error,
            "gender_shuffle_error": shuffle_gender_error,
            "gender_shuffle_dist": gender_shuffle_dist,
            "gender_shuffle_stats": gender_shuffle_stats,

            # Fig. 5c: controlled emotion classification
            "emotion_error": emotion_error,
            "emotion_shuffle_error": shuffle_emotion_error,
            "emotion_shuffle_dist": emotion_shuffle_dist,
            "emotion_shuffle_stats": emotion_shuffle_stats,

            # Cross-validation / balancing bookkeeping
            "cv_folds_requested": self.cv_folds,
            "bw_class_n": bw_class_n,
            "within_pair_class_n": within_pair_class_n,
            "gender_pair_class_n": gender_pair_class_n,
            "emotion_pair_class_n": emotion_pair_class_n,

            # Fig. 5b within-task pairwise analyses
            "true_within_pair_errors": true_within_errors,
            "within_pair_shuffle_errors": shuffle_within_errors,
            "within_pair_shuffle_stats": within_shuffle_stats_list,

            # Fig. 5c gender matched comparisons
            "gender_pair_errors": gender_pair_errors,
            "gender_pair_shuffle_errors": gender_pair_shuffle_errors,
            "gender_pair_shuffle_stats": gender_pair_shuffle_stats,

            # Fig. 5c emotion matched comparisons
            "emotion_pair_errors": emotion_pair_errors,
            "emotion_pair_shuffle_errors": emotion_pair_shuffle_errors,
            "emotion_pair_shuffle_stats": emotion_pair_shuffle_stats,
        }

        with open(
                self.save_path,
                "wb",
        ) as path:
            pickle.dump(
                lda_info,
                path,
                protocol=4,
            )

        return lda_info


class LatentSeparation:
    """Quantify latent trajectory lengths, distances, and RT associations."""

    def __init__(self, expt_stats, min_N=5, post_on_dur=2000):
        self.expt_stats = expt_stats
        self.min_N = min_N
        self.t0_ind = expt_stats.n_pre
        self.t_off_ind = (
                expt_stats.n_pre
                + 1
                + np.round(
            post_on_dur / expt_stats.step
        ).astype("int")
        )
        self._get_task_centroids()
        self.combos = get_stimulus_combos()

    def _get_task_centroids(self):
        # Calculate latent state centroids at the mean RT for each task
        # Trial type 0: repeat (gender -> gender)
        repeat_gender = {"task_cue": 0, "prev_task_cue": 0}
        self.repeat_gender_inds = self.expt_stats.select(**repeat_gender)
        self.repeat_gender_centroid = self._get_centroid(self.repeat_gender_inds)
        # Trial type 1: repeat (emotion -> emotion)
        repeat_emotion = {"task_cue": 1, "prev_task_cue": 1}
        self.repeat_emotion_inds = self.expt_stats.select(**repeat_emotion)
        self.repeat_emotion_centroid = self._get_centroid(self.repeat_emotion_inds)
        # Trial type 2: switch (emotion -> gender)
        switch_e2g = {"task_cue": 0, "prev_task_cue": 1}
        self.switch_e2g_inds = self.expt_stats.select(**switch_e2g)
        self.switch_e2g_centroid = self._get_centroid(self.switch_e2g_inds)
        # Trial type 3: switch (gender -> emotion)
        switch_g2e = {"task_cue": 1, "prev_task_cue": 0}
        self.switch_g2e_inds = self.expt_stats.select(**switch_g2e)
        self.switch_g2e_centroid = self._get_centroid(self.switch_g2e_inds)

    def _get_centroid(self, trial_inds):
        # Centroid calculated at the mean RT
        z = self.expt_stats.windowed["latents"][:, trial_inds, :]
        return np.mean(np.squeeze(z[self.t0_ind, :, :]), 0)

    def _get_trial_distances(self, trial_inds, centroid):
        z = self.expt_stats.windowed["latents"][self.t0_ind, trial_inds, :]
        return np.linalg.norm(z - centroid, axis=1, ord=2)

    def _get_hypercube_norm(self, z):
        # Calculate the volume of the hypercube defined by the
        # max/min of each latent state variable,
        # return the 16th root (= z_dim).
        z_dim = z.shape[2]
        mins = np.amin(z, axis=(0, 1))
        maxs = np.amax(z, axis=(0, 1))
        ranges = maxs - mins
        return np.prod(ranges ** (1 / z_dim))

    def analyze(self):
        """Analyze switch-trial onset-distance/RT relationships."""
        stats = {}

        # Get correlation between distance to task centroid and RT
        all_r = []
        num_low_N = 0
        num_constant = 0
        for c in self.combos:
            stim_gender, stim_emotion, cue = c

            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": 1 - cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            if cue == 0:  # gender task
                c_centroid = self.repeat_gender_centroid
            else:  # emotion task
                c_centroid = self.repeat_emotion_centroid
            this_dists = self._get_trial_distances(trial_inds, c_centroid)
            rts = self.expt_stats.df["mrt_ms"][trial_inds]
            if len(trial_inds) >= self.min_N:
                combo_r, _ = pearsonr(this_dists, rts)  # Pearson correlation
                if np.isnan(
                        combo_r
                ):  # Happens when RT array is constant (very rare)
                    num_constant += 1
                else:
                    all_r.append(combo_r)
            else:
                num_low_N += 1
        stats["all_r"] = all_r
        stats["mean_r"] = np.mean(all_r)
        stats["num_low_N"] = num_low_N
        stats["num_constant"] = num_constant

        # Get distance between task centroids
        norm_factor = self._get_hypercube_norm(self.expt_stats.latents)
        centroid_dist = np.linalg.norm(
            self.switch_g2e_centroid - self.repeat_emotion_centroid, ord=2
        )
        stats["normed_centroid_dist"] = centroid_dist / norm_factor

        return stats

    def analyze_repeat(self):
        """Analyze repeat-trial trajectory-length/RT relationships."""
        stats = {}
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()

        # Get correlation between distance to task centroid and RT
        all_r = []
        num_low_N = 0
        num_constant = 0
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[self.t0_ind:t_idx_rt, idx, :])
            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            rts = self.expt_stats.df["mrt_ms"][trial_inds]
            if len(trial_inds) >= self.min_N:
                combo_r, _ = pearsonr(this_dists, rts)  # Pearson correlation
                if np.isnan(
                        combo_r
                ):  # Happens when RT array is constant (very rare)
                    num_constant += 1
                else:
                    all_r.append(combo_r)
            else:
                num_low_N += 1
        stats["all_r"] = all_r
        stats["mean_r"] = np.mean(all_r)
        stats["num_low_N"] = num_low_N
        stats["num_constant"] = num_constant

        # Get distance between task centroids
        norm_factor = self._get_hypercube_norm(self.expt_stats.latents)
        centroid_dist = np.linalg.norm(
            self.repeat_gender_centroid - self.repeat_emotion_centroid, ord=2
        )
        stats["normed_centroid_dist"] = centroid_dist / norm_factor

        return stats

    def analyze_latent_dist(self):
        """Analyze switch-trial trajectory-length/RT relationships."""
        stats = {}
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()

        # Get correlation between distance to task centroid and RT
        all_r = []
        num_low_N = 0
        num_constant = 0
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": 1 - cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[self.t0_ind:t_idx_rt, idx, :])
            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            rts = self.expt_stats.df["mrt_ms"][trial_inds]
            if len(trial_inds) >= self.min_N:
                combo_r, _ = pearsonr(this_dists, rts)  # Pearson correlation
                if np.isnan(
                        combo_r
                ):  # Happens when RT array is constant (very rare)
                    num_constant += 1
                else:
                    all_r.append(combo_r)
            else:
                num_low_N += 1
        stats["all_r"] = all_r
        stats["mean_r"] = np.mean(all_r)
        stats["num_low_N"] = num_low_N
        stats["num_constant"] = num_constant

        # Get distance between task centroids
        norm_factor = self._get_hypercube_norm(self.expt_stats.latents)
        centroid_dist = np.linalg.norm(
            self.repeat_gender_centroid - self.repeat_emotion_centroid, ord=2
        )
        stats["normed_centroid_dist"] = centroid_dist / norm_factor

        return stats

    def analyze_latent_dist_switch(self, current_task: int):
        """Analyze one switch direction and its switch-repeat onset distance."""
        stats = {}
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()
        norm_factor = self._get_hypercube_norm(self.expt_stats.latents)

        # Get correlation between distance to task centroid and RT
        all_r = []
        num_low_N = 0
        num_constant = 0
        all_normed_dist = []
        all_extent_weighted_dist = []
        all_euc_dist = []
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            if cue != current_task:  # Depending on the current task, skip specific trials
                continue
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": 1 - cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[self.t0_ind:t_idx_rt, idx, :])
            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            rts = self.expt_stats.df["mrt_ms"][trial_inds]
            if len(trial_inds) >= self.min_N:
                combo_r, _ = pearsonr(this_dists, rts)  # Pearson correlation
                if np.isnan(
                        combo_r
                ):  # Happens when RT array is constant (very rare)
                    num_constant += 1
                else:
                    all_r.append(combo_r)
            else:
                num_low_N += 1
            # Compute condition-level normed distance
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            onset_switch = np.mean(np.squeeze(latent_traj[self.t0_ind, :, :]), 0)
            filter_repeat = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds_repeat = self.expt_stats.select(**filter_repeat)
            latent_traj_repeat = self.expt_stats.windowed["latents"][:, trial_inds_repeat, :]
            onset_repeat = np.mean(np.squeeze(latent_traj_repeat[self.t0_ind, :, :]), 0)
            centroid_dist = np.linalg.norm(
                onset_switch - onset_repeat, ord=2
            )
            all_normed_dist.append(centroid_dist / norm_factor)
            all_extent_weighted_dist.append(centroid_dist * norm_factor)
            all_euc_dist.append(centroid_dist)
        stats["all_r"] = all_r
        stats["mean_r"] = np.mean(all_r)
        stats["num_low_N"] = num_low_N
        stats["num_constant"] = num_constant
        stats["normed_centroid_dist"] = np.mean(all_normed_dist)
        stats["latent_scale_weighted_centroid_dist"] = np.mean(all_extent_weighted_dist)
        stats["euc_dist"] = np.mean(all_euc_dist)

        return stats

    def analyze_latent_dist_repeat(self, current_task: int):
        """Analyze repeat trials for one current task."""
        stats = {}
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()

        # Get correlation between distance to task centroid and RT
        all_r = []
        num_low_N = 0
        num_constant = 0
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            if cue != current_task:  # Depending on the current task, skip specific trials
                continue
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[self.t0_ind:t_idx_rt, idx, :])
            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            rts = self.expt_stats.df["mrt_ms"][trial_inds]
            if len(trial_inds) >= self.min_N:
                combo_r, _ = pearsonr(this_dists, rts)  # Pearson correlation
                if np.isnan(
                        combo_r
                ):  # Happens when RT array is constant (very rare)
                    num_constant += 1
                else:
                    all_r.append(combo_r)
            else:
                num_low_N += 1
        stats["all_r"] = all_r
        stats["mean_r"] = np.mean(all_r)
        stats["num_low_N"] = num_low_N
        stats["num_constant"] = num_constant

        # Get distance between task centroids
        norm_factor = self._get_hypercube_norm(self.expt_stats.latents)
        if current_task == 1:  # emotion task
            centroid_dist = np.linalg.norm(
                self.switch_g2e_centroid - self.repeat_emotion_centroid, ord=2
            )
        else:
            centroid_dist = np.linalg.norm(
                self.switch_e2g_centroid - self.repeat_gender_centroid, ord=2
            )
        stats["normed_centroid_dist"] = centroid_dist / norm_factor

        return stats

    def get_org_length_full_repeat(self, current_task: int):
        """Return native-space onset-to-RT path lengths for repeat trials."""
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()
        list_traj = []
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            if cue != current_task:
                continue
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[self.t0_ind:t_idx_rt, idx, :])
            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            list_traj.append(this_dists)

        return list_traj

    def get_org_length_full_switch(self, current_task: int):
        """Return native-space onset-to-RT path lengths for switch trials."""
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()
        list_traj = []
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            if cue != current_task:
                continue
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": 1 - cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[self.t0_ind:t_idx_rt, idx, :])

            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            list_traj.append(this_dists)

        return list_traj

    def get_org_length_starting_specific_repeat(self, current_task: int, start_t: float):
        """Return repeat-trial path lengths from ``start_t`` to RT."""
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()
        list_traj = []
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            if cue != current_task:
                continue
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            start_t_ind = np.round(start_t / self.expt_stats.step).astype("int")
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[start_t_ind:t_idx_rt, idx, :])
            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            list_traj.append(this_dists)

        return list_traj

    def get_org_length_starting_specific_switch(self, current_task: int, start_t: float):
        """Return switch-trial path lengths from ``start_t`` to RT."""
        m_rts = self.expt_stats.df["mrt_ms"].to_numpy()
        list_traj = []
        for c in self.combos:
            stim_gender, stim_emotion, cue = c
            if cue != current_task:
                continue
            filter = {
                "stim_gender": stim_gender,
                "stim_emotion": stim_emotion,
                "task_cue": cue,
                "prev_task_cue": 1 - cue,
                "mcorrect": 1,
                "m_prev_correct": 1,
            }
            trial_inds = self.expt_stats.select(**filter)
            rts = np.round(m_rts[trial_inds] / self.expt_stats.step).astype("int")
            t_ind_rt = rts + self.expt_stats.n_pre
            start_t_ind = np.round(start_t / self.expt_stats.step).astype("int")
            latent_traj = self.expt_stats.windowed["latents"][:, trial_inds, :]
            latent_vary_traj = []
            for idx, t_idx_rt in enumerate(t_ind_rt):
                latent_vary_traj.append(latent_traj[start_t_ind:t_idx_rt, idx, :])
            this_dists = self._compute_trajectory_distance(latent_vary_traj)
            list_traj.append(this_dists)

        return list_traj

    def _compute_cumulative_trajectory_lengths(self, z_traj):
        """Return cumulative Euclidean path length for one trajectory."""
        diffs = z_traj[1:] - z_traj[:-1]
        segment_lengths = np.linalg.norm(
            diffs,
            axis=1,
        )
        return np.sum(segment_lengths)

    def _compute_trajectory_distance(self, z_traj):
        return [self._compute_cumulative_trajectory_lengths(z) for z in z_traj]
