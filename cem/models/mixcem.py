import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics
from cem.models.cbm import _get_last_linear
from torchvision.models import resnet50
import torch.nn as nn
import cem.train.utils as utils
from cem.models.intcbm import IntAwareConceptEmbeddingModel
from cem.metrics.accs import compute_accuracy
from cem.models.cbm import AverageMeter, batch_diversity


class MixCEM(IntAwareConceptEmbeddingModel):
    """
    Mixture of Concept Embeddings Model (MixCEM) as proposed by
    Espinosa Zarlenga et al. at ICML 2025 (https://arxiv.org/abs/2504.17921).
    """

    def __init__(
            self,
            n_concepts,
            n_tasks,
            emb_size=16,
            training_intervention_prob=0.25,
            embedding_activation="leakyrelu",
            concept_loss_weight=1,

            c2y_model=None,
            c2y_layers=None,
            c_extractor_arch=utils.wrap_pretrained_model(resnet50),
            output_latent=False,

            optimizer="adam",
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=4e-05,
            lr_scheduler_factor=0.1,
            lr_scheduler_patience=10,
            weight_loss=None,
            task_class_weights=None,

            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_policy=None,
            output_interventions=False,

            top_k_accuracy=None,
            sample_c_preds=True,
            beta_max=10,
            # Experimental/debugging arguments
            intervention_discount=1,
            include_only_last_trajectory_loss=True,
            task_loss_weight=1,
            intervention_task_loss_weight=1,

            ##################################
            # New MixCEM-specific arguments
            #################################
            ood_dropout_prob=0.5,  # Probability of random dynamic component drop in training (λ_drop in paper)
            all_intervened_loss_weight=1,  # Strength of prior error (λ_p in paper)
            initial_concept_embeddings=None,
            fixed_embeddings=False,
            temperature=1,
            kl_ratio=0.1,
            # Monte carlo stuff
            montecarlo_test_tries=50,  # Number of MC trials to use during inference
            deterministic=False,
            montecarlo_train_tries=1,
            output_uncertainty=False,
            hard_selection_value=None,

            ##################################
            # IntCEM-specific arguments (use this only for IntMixCEM, that is
            # MixCEM's intervention-aware version)
            #################################

            # Intervention-aware hyperparameters (NO intervention-aware loss
            # is used by default)
            intervention_task_discount=1,
            intervention_weight=0,
            concept_map=None,
            use_concept_groups=True,
            rollout_init_steps=0,
            int_model_layers=None,
            int_model_use_bn=False,
            num_rollouts=1,
            max_horizon=1,
            initial_horizon=2,
            horizon_rate=1,
            topk_concepts_path=None,
    ):
        self.temperature = temperature
        self._context_scale_factors = None
        self.all_intervened_loss_weight = all_intervened_loss_weight
        self.hard_selection_value = hard_selection_value
        self._construct_c2y_model = False
        bottleneck_size = emb_size * n_concepts
        self.montecarlo_train_tries = montecarlo_train_tries
        self.montecarlo_test_tries = montecarlo_test_tries
        self.deterministic = deterministic
        self.ood_dropout_prob = ood_dropout_prob
        self.output_uncertainty = output_uncertainty
        self.kl_ratio = kl_ratio
        self._mixed_stds = None
        self.topk_concepts_path = topk_concepts_path
        self.topk_concepts_of_class = None
        if topk_concepts_path is not None:
            loaded = torch.load(topk_concepts_path, map_location="cpu")
            if isinstance(loaded, dict):
                rows = []
                topk_len = None
                for cls in range(n_tasks):
                    v = loaded.get(cls, None)
                    if v is None:
                        v = loaded.get(str(cls), None)
                    if v is None:
                        raise ValueError(
                            f"Missing topk indices for class {cls} in {topk_concepts_path}"
                        )
                    t = torch.as_tensor(v, dtype=torch.long).reshape(-1)
                    if topk_len is None:
                        topk_len = int(t.numel())
                    if int(t.numel()) != topk_len:
                        raise ValueError(
                            f"Inconsistent topk sizes in {topk_concepts_path}: "
                            f"expected {topk_len} but class {cls} has {int(t.numel())}"
                        )
                    rows.append(t)
                self.topk_concepts_of_class = torch.stack(rows, dim=0)  # (n_tasks, topk)
            else:
                t = torch.as_tensor(loaded, dtype=torch.long)
                if t.dim() != 2:
                    raise ValueError(
                        f"Expected topk_concepts_of_class to be 2D (n_tasks, topk) but got {tuple(t.shape)}"
                    )
                self.topk_concepts_of_class = t

        super(MixCEM, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            concept_loss_weight=concept_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            intervention_task_discount=intervention_task_discount,
            intervention_weight=intervention_weight,
            concept_map=concept_map,
            use_concept_groups=use_concept_groups,
            rollout_init_steps=rollout_init_steps,
            int_model_layers=int_model_layers,
            int_model_use_bn=int_model_use_bn,
            num_rollouts=num_rollouts,
            max_horizon=max_horizon,
            initial_horizon=initial_horizon,
            horizon_rate=horizon_rate,
            intervention_discount=intervention_discount,
            include_only_last_trajectory_loss=include_only_last_trajectory_loss,
            task_loss_weight=task_loss_weight,
            intervention_task_loss_weight=intervention_task_loss_weight,
            bottleneck_size=bottleneck_size,
        )

        if not hasattr(self, "N1"):
            self.register_buffer("N1", torch.zeros(self.n_concepts))
        if not hasattr(self, "N0"):
            self.register_buffer("N0", torch.zeros(self.n_concepts))
        if not hasattr(self, "ema"):
            self.ema = 0.9
        # Let's generate the global embeddings we will use
        if (initial_concept_embeddings is False) or (
                initial_concept_embeddings is None
        ):
            initial_concept_embeddings = torch.normal(
                torch.zeros(self.n_concepts, 2, emb_size),
                torch.ones(self.n_concepts, 2, emb_size),
            )
        else:
            if isinstance(initial_concept_embeddings, np.ndarray):
                initial_concept_embeddings = torch.FloatTensor(
                    initial_concept_embeddings
                )
            emb_size = initial_concept_embeddings.shape[-1]
        self.concept_embeddings = torch.nn.Parameter(
            initial_concept_embeddings,
            requires_grad=(not fixed_embeddings),
        )

        if c2y_model is None:
            # Else we construct it here directly
            units = [
                        n_concepts * emb_size
                    ] + (c2y_layers or []) + [n_tasks]
            layers = [torch.nn.Flatten(1, -1)]
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i - 1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)
        else:
            self.c2y_model = c2y_model

        self.global_c2y_model = self.c2y_model

        self.ood_dropout_prob = ood_dropout_prob
        self.global_concept_context_generators = torch.nn.Linear(
            list(
                self.pre_concept_model.modules()
            )[-1].out_features,
            self.emb_size,
        )
        self.concept_scales = torch.nn.Parameter(
            torch.rand(n_concepts),
            requires_grad=True,
        )
        self.ood_uncertainty_thresh = torch.nn.Parameter(
            torch.zeros(1),
            requires_grad=False,
        )
        self.concept_platt_scales = torch.nn.Parameter(
            torch.ones(n_concepts),
            requires_grad=False,
        )
        self.concept_platt_biases = torch.nn.Parameter(
            torch.zeros(n_concepts),
            requires_grad=False,
        )
        self.logit_temperatures = torch.nn.Parameter(
            torch.ones(n_concepts, n_tasks),
            requires_grad=False,
        )

        # MixCEM uses a Beta distribution for each concept probability.
        # We therefore need TWO generators per concept: one for beta_a and one
        # for beta_b. We keep the shared/non-shared behavior consistent with
        # `shared_prob_gen`.
        beta_in_features = 2 * self.emb_size
        self.concept_beta_generators = torch.nn.ModuleList()
        if self.shared_prob_gen:
            self.concept_beta_generators.append(
                torch.nn.ModuleList([
                    torch.nn.Linear(beta_in_features, 1),
                    torch.nn.Linear(beta_in_features, 1),
                ])
            )
        else:
            for _ in range(self.n_concepts):
                self.concept_beta_generators.append(
                    torch.nn.ModuleList([
                        torch.nn.Linear(beta_in_features, 1),
                        torch.nn.Linear(beta_in_features, 1),
                    ])
                )
        self.sample_c_preds = sample_c_preds
        self.beta_max = beta_max
        self._init_meters()

    def _init_meters(self):
        keys = [
            "prior_mean",
            "posterior_mean",
            "task_loss",
            "kl",
            "Em",
            "Ea",
            "Eb",
            "Ea_over_b",
            "diversity",
            "concept_label_dot",
        ]
        self.train_meters = {k: AverageMeter() for k in keys}
        self.val_meters = {k: AverageMeter() for k in keys}

    def _uncertainty_based_context_addition(self, concept_probs, temperature=1):
        # We only select to add a context when the uncertainty is far from the extremes
        # 当不确定性大（信息量小）的时候，用context（residual）来补充概念表示
        entr = (
                -concept_probs * torch.log2(concept_probs + 1e-6) -
                (1 - concept_probs) * torch.log2(1 - concept_probs + 1e-6)
        )
        return self.temperature * (1 - entr)

    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        outputs = []

        if bottleneck.shape[-1] == 2:
            # Then no test time sampling was done!! So let's just use the
            # normal mixed bottleneck. This will always be the first
            # trial output
            outputs.append(
                self.logit_temperatures[0, 0] *
                self.c2y_model(bottleneck[:, :, :, 0])
            )
        else:
            for trial_idx in range(2, bottleneck.shape[-1]):
                out_vals = self.c2y_model(bottleneck[:, :, :, trial_idx])
                out_vals = self.logit_temperatures[0, 0] * out_vals
                outputs.append(out_vals.unsqueeze(-1))
            outputs = torch.concat(outputs, dim=-1)
        self._mixed_stds = torch.std(outputs, dim=-1)
        outputs = torch.mean(outputs, dim=-1)
        return outputs

    def _construct_c2y_input(
            self,
            pos_embeddings,
            neg_embeddings,
            probs,
            **task_loss_kwargs,
    ):
        # We will generate several versions of the bottleneck with different
        # masks.Then, downstream in the line with _predict_labels, we will
        # unpack them, make a label prediction, and compute the mean and
        # variance of all samples
        extra_scale = 1
        if self.deterministic or (self.hard_selection_value is not None):
            n_trials = 1
        elif not self.training:
            if self.montecarlo_test_tries == 0:
                # Then we will interpret this as a normal dropout rescaling
                # during inference
                n_trials = 1
                extra_scale = (
                    self.ood_dropout_prob if self.ood_dropout_prob > 0
                    else 1
                )
            else:
                n_trials = max(self.montecarlo_test_tries, 0)
        else:
            n_trials = max(self.montecarlo_train_tries, 0)

        global_pos_embeddings = pos_embeddings[:, :, :self.emb_size]
        contextual_pos_embeddings = pos_embeddings[:, :, self.emb_size:]
        contextual_pos_embeddings = (
                contextual_pos_embeddings *
                self._context_scale_factors.unsqueeze(-1)
        )
        global_neg_embeddings = neg_embeddings[:, :, :self.emb_size]
        contextual_neg_embeddings = neg_embeddings[:, :, self.emb_size:]
        contextual_neg_embeddings = (
                contextual_neg_embeddings *
                self._context_scale_factors.unsqueeze(-1)
        )
        bottlenecks = []
        # The first two elements of the array will always be the contextual
        # mixed embedding followed by just the one using the global embeddings
        combined_pos_embs = global_pos_embeddings + \
                            extra_scale * contextual_pos_embeddings
        combined_neg_embs = global_neg_embeddings + \
                            extra_scale * contextual_neg_embeddings
        new_bottleneck = (
                combined_pos_embs * torch.unsqueeze(probs, dim=-1) + (
                combined_neg_embs * (
                1 - torch.unsqueeze(probs, dim=-1)
        )
        )
        )
        bottlenecks.append(new_bottleneck.unsqueeze(-1))
        new_bottleneck = (
                global_pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                global_neg_embeddings * (
                1 - torch.unsqueeze(probs, dim=-1)
        )
        )
        )
        bottlenecks.append(new_bottleneck.unsqueeze(-1))
        for _ in range(n_trials):
            if self.hard_selection_value is not None:
                context_selected = (1 - self.hard_selection_value) * torch.ones(
                    global_pos_embeddings.shape[0], self.n_concepts, 1
                ).to(global_pos_embeddings.device)
            elif self.deterministic:
                context_selected = torch.ones(
                    global_pos_embeddings.shape[0], self.n_concepts, 1
                ).to(global_pos_embeddings.device)
            else:
                # Then we generate a simple random mask
                dropout_prob = self.ood_dropout_prob
                context_selected = torch.bernoulli(
                    torch.ones(
                        global_pos_embeddings.shape[0],
                        self.n_concepts,
                        1,
                    ) * (1 - dropout_prob)
                ).to(global_pos_embeddings.device)
            combined_pos_embs = global_pos_embeddings + (
                    context_selected * contextual_pos_embeddings
            )
            combined_neg_embs = global_neg_embeddings + (
                    context_selected * contextual_neg_embeddings
            )
            new_bottleneck = (
                    combined_pos_embs * torch.unsqueeze(probs, dim=-1) + (
                    combined_neg_embs * (
                    1 - torch.unsqueeze(probs, dim=-1)
            )
            )
            )
            bottlenecks.append(new_bottleneck.unsqueeze(-1))
        return torch.concat(bottlenecks, dim=-1)

    def _construct_rank_model_input(self, bottleneck, prev_interventions):
        # We always use the dynamic + global embeddings
        bottleneck = bottleneck[:, :, :, 0]
        cat_inputs = [
            bottleneck.reshape(bottleneck.shape[0], -1),
            prev_interventions,
        ]
        return torch.concat(
            cat_inputs,
            dim=-1,
        )

    def _new_tail_results(
            self,
            x=None,
            c=None,
            y=None,
            c_sem=None,
            bottleneck=None,
            y_pred=None,
    ):
        tail_results = []
        if (
                (self._mixed_stds is not None) and
                (self.output_uncertainty)
        ):
            tail_results.append(self._mixed_stds)
            self._mixed_stds = None
        return tail_results

    def _generate_dynamic_concept(self, pre_c, concept_idx):
        context = self.concept_context_generators[concept_idx](pre_c)
        return context

    def _generate_concept_embeddings(
            self,
            x,
            latent=None,
            training=False,
    ):
        extra_outputs = {}
        if latent is None:
            pre_c = self.pre_concept_model(x)  # B, 128
            global_emb_center = self.global_concept_context_generators(pre_c)  # B, 128 _generate_dynamic_context
            dynamic_contexts = []
            for concept_idx in range(self.n_concepts):  # 从image_features生成每个concept的dynamic context
                dynamic_context = self._generate_dynamic_concept(
                    pre_c,
                    concept_idx=concept_idx,
                )
                dynamic_contexts.append(torch.unsqueeze(dynamic_context, dim=1))
            dynamic_contexts = torch.cat(dynamic_contexts, axis=1)  # B, num_concepts, latent_dim * 2
            self._dynamic_context = dynamic_contexts

            global_context_pos = \
                self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
                    pre_c.shape[0],
                    -1,
                    -1,
                )
            global_context_neg = \
                self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
                    pre_c.shape[0],
                    -1,
                    -1,
                )
            global_contexts = torch.concat(
                [global_context_pos, global_context_neg],
                dim=-1,
            )  # B, num_concepts, latent_dim * 2
            latent = dynamic_contexts, global_contexts
        else:
            dynamic_contexts, global_contexts = latent

        # Now we can compute all the probabilites!
        c_sem = []
        beta_a_vals = []
        beta_b_vals = []
        for concept_idx in range(self.n_concepts):
            if self.shared_prob_gen:  # 为所有概念生成相同的先验a, b形成Beta(a, b)
                beta_a_gen = self.concept_beta_generators[0][0]
                beta_b_gen = self.concept_beta_generators[0][1]
            else:                    # 为所有概念生成不同的先验a, b形成Beta(a, b)
                beta_a_gen = self.concept_beta_generators[concept_idx][0]
                beta_b_gen = self.concept_beta_generators[concept_idx][1]
            if self.hard_selection_value is None:
                beta_a = beta_a_gen(
                    global_contexts[:, concept_idx, :] +
                    dynamic_contexts[:, concept_idx, :]
                )  # B, latent_dim * 2 -> B, 1
                beta_b = beta_b_gen(
                    global_contexts[:, concept_idx, :] +
                    dynamic_contexts[:, concept_idx, :]
                )  # B, latent_dim * 2 -> B, 1
            else:
                # 固定比例缩放dynamic，（global + (1-hard)dynamic）
                beta_a = beta_a_gen(
                    global_contexts[:, concept_idx, :] +
                    (
                            (1 - self.hard_selection_value) *
                            dynamic_contexts[:, concept_idx, :]
                    )
                )
                beta_b = beta_b_gen(
                    global_contexts[:, concept_idx, :] +
                    (
                            (1 - self.hard_selection_value) *
                            dynamic_contexts[:, concept_idx, :]
                    )
                )
            # 根据Beta分布采样生成概率
            # Beta distribution requires strictly positive concentration params.
            beta_a = F.softplus(beta_a) + 1e-4
            beta_b = F.softplus(beta_b) + 1e-4
            beta_a = torch.clamp(beta_a, max=self.beta_max)
            beta_b = torch.clamp(beta_b, max=self.beta_max)
            beta_a_vals.append(beta_a)
            beta_b_vals.append(beta_b)
            prior_theta = torch.distributions.Beta(beta_a, beta_b)
            prob = prior_theta.rsample()
            c_sem.append(prob)

        # record beta priors and kl divergence
        beta_a_vals = torch.cat(beta_a_vals, dim=-1) # B, n_concepts
        beta_b_vals = torch.cat(beta_b_vals, dim=-1) # B, n_concepts
        self._last_beta_a = beta_a_vals
        self._last_beta_b = beta_b_vals

        c_sem = torch.cat(c_sem, axis=-1) # B, n_concepts c_probs
        # Sanity check + clamp for downstream entropy computations.
        if not torch.isfinite(c_sem).all():
            raise ValueError("Non-finite values found in c_sem (concept probabilities).")
        c_sem = torch.clamp(c_sem, min=1e-6, max=1 - 1e-6)
        self._context_scale_factors = self._uncertainty_based_context_addition(
            concept_probs=c_sem,
            temperature=self.temperature,
        )
        if self.hard_selection_value is not None:
            self._context_scale_factors = \
                1 - self.hard_selection_value * torch.ones_like(
                    self._context_scale_factors
                )

        pos_embeddings = torch.concat(
            [
                global_contexts[:, :, :self.emb_size],
                dynamic_contexts[:, :, :self.emb_size],
            ],
            dim=-1
        )
        neg_embeddings = torch.concat(
            [
                global_contexts[:, :, self.emb_size:],
                dynamic_contexts[:, :, self.emb_size:],
            ],
            dim=-1
        )
        return c_sem, pos_embeddings, neg_embeddings, extra_outputs

    def _extra_losses(
            self,
            x,
            y,
            c,
            y_pred,
            c_sem,
            c_pred,
            competencies=None,
            prev_interventions=None,
    ):
        loss = 0.0
        if self.all_intervened_loss_weight != 0:
            global_context_pos = \
                self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
                    c.shape[0],
                    -1,
                    -1,
                )  # self.concept_embeddings [num_concepts, 2, latent_dim] -> global_context_pos [B, num_concepts, latent_dim]
            global_context_neg = \
                self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
                    c.shape[0],
                    -1,
                    -1,
                )
            new_bottleneck = (
                    global_context_pos * torch.unsqueeze(c, dim=-1) + (
                    global_context_neg * (
                    1 - torch.unsqueeze(c, dim=-1)
            )
            )
            )  # mix 正负融合 global_context_pos [B, num_concepts, latent_dim]
            # compute task loss
            new_y_logits = \
                self.logit_temperatures[0, 0] * self.c2y_model(new_bottleneck)
            loss += self.all_intervened_loss_weight * self.loss_task(
                (
                    new_y_logits if new_y_logits.shape[-1] > 1
                    else new_y_logits.reshape(-1)
                ),
                y,
            )
        return loss

    def _compute_logging_stats(
        self,
        pre_c,
        beta_a,
        beta_b,
        c_pred,
        y,
        task_loss,
        kl_loss,
        meters,
        use_precise_posterior=False,
    ):
        """
        beta_a, beta_b: (B, K)
        c_pred: (B, K)
        y: (B,)
        meters: dict of AverageMeter
        """
        B, K = c_pred.shape

        # ---------- 1. prior / posterior mean ----------
        a = beta_a.mean(dim=0)  # K
        b = beta_b.mean(dim=0)  # K
        prior_mean = (a / (a + b)).mean().item()

        if use_precise_posterior:
            post_mean = ((a + self.N1) / (a + b + self.N1 + self.N0)).mean().item()
        else:
            # batch posterior
            m = (c_pred >= 0.5).float()
            N1 = m.sum(dim=0)
            N0 = B - N1
            post_mean = ((a + N1) / (a + b + N1 + N0)).mean().item()

        # ---------- 2. E[m] ----------
        Em = (c_pred >= 0.5).float().mean().item()

        # ---------- 3. Beta parameter expectations ----------
        Ea = a.mean().item()
        Eb = b.mean().item()
        Ea_over_b = (a / (a+b + 1e-6)).mean().item()

        # ---------- 4. batch diversity ----------
        diversity, total_diversity = batch_diversity(c_pred)


        # ---------- 5. topk_concept_cosine_similarity cosine(c_i, Wgk) ----------
        topk_concept_cosine_similarity = None
        if (self.topk_concepts_of_class is not None) and (pre_c is not None):
            # (B, topk)
            y_to_topk_concept_indices = self.topk_concepts_of_class.to(y.device)[y.long()]

            # Prefer x2c_model.fc.weight as requested; fallback to last Linear.
            x2c_last = getattr(self.x2c_model, "fc", None)
            print("x2c fc", x2c_last)
            if not isinstance(x2c_last, nn.Linear):
                x2c_last = _get_last_linear(self.x2c_model)

            if (x2c_last is not None) and hasattr(x2c_last, "weight"):
                W_full = x2c_last.weight  # (out_dim, F)
                W_c = W_full[: self.n_concepts]
                W_topk_c = W_c[y_to_topk_concept_indices]  # (B, topk, F)

                # Vectorized cosine similarity and average score in [0, 1].
                W_hat = F.normalize(W_topk_c, dim=-1)
                pre_hat = F.normalize(pre_c, dim=-1).unsqueeze(1)  # (B, 1, F)
                cos_sim = (W_hat * pre_hat).sum(dim=-1)  # (B, topk)
                topk_concept_cosine_similarity = ((cos_sim + 1.0) / 2.0).mean()

        else:
            W = self.c2y_model[1].weight    # (n_tasks, n_concepts x F)
            W_y = W[y]                      # y: (B,) -> (B, n_concepts x F)
            W_y = F.normalize(W_y, dim=1)
            pre_c = F.normalize(pre_c.view(B, -1), dim=1)
            topk_concept_cosine_similarity = torch.einsum("bk,bk->b", pre_c, W_y
                                                              ).mean().item()  # dot product row by row <c_i, W[y_i, :]>

        meters["prior_mean"].update(prior_mean, B)
        meters["posterior_mean"].update(post_mean, B)
        meters["task_loss"].update(task_loss.item(), B)
        meters["kl"].update(kl_loss.item(), B)
        meters["Em"].update(Em, B)
        meters["Ea"].update(Ea, B)
        meters["Eb"].update(Eb, B)
        meters["Ea_over_b"].update(Ea_over_b, B)
        meters["diversity"].update(float(total_diversity.item()), B)
        meters["concept_label_dot"].update(topk_concept_cosine_similarity, B)

    def _run_step(
            self,
            batch,
            batch_idx,
            train=False,
            intervention_idxs=None,
            use_precise_posterior=False,
    ):
        """
        MixCEM-specific implementation of _run_step.
        Adapted from CBM._run_step to handle MixCEM's concept embeddings and Beta distributions.
        """
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(batch)

        # Forward pass to get concept predictions and task predictions
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
        )
        c_sem = outputs[0]  # Concept probabilities from Beta distribution
        bottleneck = outputs[1]  # Mixed concept embeddings
        y_logits = outputs[2]  # Task predictions
        self._last_pre_c = self.c2y_model[0](bottleneck[:, :, :, 2])

        # Compute concept loss
        if self.concept_loss_weight != 0:
            concept_loss = self.loss_concept(c_sem, c)  # BCELoss on concept predictions
            concept_loss_scalar = concept_loss.detach()
        else:
            concept_loss = torch.tensor(0.0, device=y_logits.device)
            concept_loss_scalar = concept_loss.item()

        # Compute task loss
        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y.long(),
            )
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = torch.tensor(0.0, device=y_logits.device)
            task_loss_scalar = task_loss.item()

        beta_a = getattr(self, "_last_beta_a", None)    # B, n_concepts
        beta_b = getattr(self, "_last_beta_b", None)    # B, n_concepts
        kl_loss = self.beta_kl(beta_a, beta_b, c_sem)   # KL divergence

        # Compute total loss with extra losses
        loss = (
                self.concept_loss_weight * concept_loss +
                self.task_loss_weight * task_loss +
                self.kl_ratio * kl_loss +
                self._extra_losses(
                    x=x,
                    y=y,
                    c=c,
                    c_sem=c_sem,
                    c_pred=bottleneck,
                    y_pred=y_logits,
                    competencies=competencies,
                    prev_interventions=prev_interventions,
                )
        )

        meters = self.train_meters if train else self.val_meters
        pre_c = getattr(self, "_last_pre_c", None)
        self._compute_logging_stats(
            pre_c=pre_c,
            beta_a=beta_a,
            beta_b=beta_b,
            c_pred=c_sem,
            y=y,
            task_loss=task_loss,
            kl_loss=kl_loss,
            meters=meters,
            use_precise_posterior=use_precise_posterior,
        )

        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
        }

        # Compute top-k accuracy if specified
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            if isinstance(self.top_k_accuracy, int):
                top_k_accuracy = list(range(1, self.top_k_accuracy))
            else:
                top_k_accuracy = self.top_k_accuracy

            for top_k_val in top_k_accuracy:
                if top_k_val:
                    y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                        y_true,
                        y_pred,
                        k=top_k_val,
                        labels=labels,
                    )
                    result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy

        return loss, result

    def validation_step(self, batch, batch_no):
        x, y, _ = self._unpack_batch(batch)
        with torch.no_grad():
            pre_c = self.pre_concept_model(x)
            dynamic_contexts = []
            for concept_idx in range(self.n_concepts):
                dynamic_context = self._generate_dynamic_concept(
                    pre_c,
                    concept_idx=concept_idx,
                )
                dynamic_contexts.append(torch.unsqueeze(dynamic_context, dim=1))
            dynamic_contexts = torch.cat(dynamic_contexts, axis=1)

            global_context_pos = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
                pre_c.shape[0],
                -1,
                -1,
            )
            global_context_neg = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
                pre_c.shape[0],
                -1,
                -1,
            )
            global_contexts = torch.concat(
                [global_context_pos, global_context_neg],
                dim=-1,
            )

            theta_vals = []
            for concept_idx in range(self.n_concepts):
                if self.shared_prob_gen:
                    beta_a_gen = self.concept_beta_generators[0][0]
                    beta_b_gen = self.concept_beta_generators[0][1]
                else:
                    beta_a_gen = self.concept_beta_generators[concept_idx][0]
                    beta_b_gen = self.concept_beta_generators[concept_idx][1]

                if self.hard_selection_value is None:
                    beta_inputs = (
                            global_contexts[:, concept_idx, :] +
                            dynamic_contexts[:, concept_idx, :]
                    )
                else:
                    beta_inputs = (
                            global_contexts[:, concept_idx, :] +
                            (1 - self.hard_selection_value) *
                            dynamic_contexts[:, concept_idx, :]
                    )

                beta_a = F.softplus(beta_a_gen(beta_inputs)) + 1e-4
                beta_b = F.softplus(beta_b_gen(beta_inputs)) + 1e-4
                beta_a = torch.clamp(beta_a, max=self.beta_max)
                beta_b = torch.clamp(beta_b, max=self.beta_max)
                theta = beta_a / (beta_a + beta_b)
                theta_vals.append(theta.reshape(-1, 1))

            c_pred = torch.cat(theta_vals, dim=-1)
            m = (c_pred >= 0.5).float()

        # EMA update
        self.N1 = self.ema * self.N1 + (1 - self.ema) * m.sum(dim=0)
        self.N0 = self.ema * self.N0 + (1 - self.ema) * (m.size(0) - m.sum(dim=0))

        _, result = self._run_step(batch, batch_no, train=False)
        self.log("val/loss", result.get("loss", 0), prog_bar=True)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = (("auc" in name))
            else:
                prog_bar = (("c_auc" in name) or ("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar)
        return {"val_" + key: val for key, val in result.items()}

    def _concept_platt_scaling(self, logits, concept_idx):
        return (
                self.concept_platt_scales[concept_idx] * logits +
                self.concept_platt_biases[concept_idx]
        )

    def freeze_global_components(self):
        self.concept_embeddings.requires_grad = False
        self.concept_scales.requires_grad = False
        for param in self.global_concept_context_generators.parameters():
            param.requires_grad = False
        for param in self.pre_concept_model.parameters():
            param.requires_grad = False

    def freeze_ood_separator(self):
        pass

    def unfreeze_ood_separator(self):
        pass

    def unfreeze_calibration_components(
            self,
            unfreeze_dynamic=True,
            unfreeze_global=True,
    ):
        self.concept_platt_scales.requires_grad = True
        self.concept_platt_biases.requires_grad = True

    def freeze_calibration_components(
            self,
            freeze_dynamic=True,
            freeze_global=True,
    ):
        self.concept_platt_scales.requires_grad = False
        self.concept_platt_biases.requires_grad = False

    def unfreeze_global_components(self):
        self.concept_embeddings.requires_grad = True
        self.concept_scales.requires_grad = True
        for param in self.global_concept_context_generators.parameters():
            param.requires_grad = True
        for param in self.pre_concept_model.parameters():
            param.requires_grad = True



