import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torchvision.models import resnet50

from cem.metrics.accs import compute_accuracy
from cem.models.cbm import ConceptBottleneckModel, AverageMeter, batch_diversity
import cem.train.utils as utils



################################################################################
## Concept Embedding Models
################################################################################


class ConceptEmbeddingModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

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
        use_concept_groups=False,

        context_gen_out_size=None,

        top_k_accuracy=None,
    ):
        """
        Constructs a Concept Embedding Model (CEM) as defined by
        Espinosa Zarlenga et al. 2022.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param float training_intervention_prob: RandInt probability. Defaults
            to 0.25.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this CEM's inputs to the latent space of the
            requested size.

        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.

        :param List[float] active_intervention_values: A list of n_concepts
            values to use when positively intervening in a given concept (i.e.,
            setting concept c_i to 1 would imply setting its corresponding
            predicted concept to active_intervention_values[i]). If not given,
            then we will assume that we use `1` for all concepts. This
            parameter is important when intervening in CEMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts.
        :param Callable[(np.ndarray, np.ndarray, np.ndarray), np.ndarray] intervention_policy:
            An optional intervention policy to be used when intervening on a
            test batch sample x (first argument), with corresponding true
            concepts c (second argument), and true labels y (third argument).
            The policy must produce as an output a list of concept indices to
            intervene (in batch form) or a batch of binary masks indicating
            which concepts we will intervene on.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        context_gen_out_size = context_gen_out_size or (2 * emb_size)
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self._intervention_idxs = None
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy
        for i in range(n_concepts):
            if embedding_activation is None:
                act_to_use = []
            elif embedding_activation == "sigmoid":
                act_to_use = [torch.nn.Sigmoid()]
            elif embedding_activation == "leakyrelu":
                act_to_use = [torch.nn.LeakyReLU()]
            elif embedding_activation == "relu":
                act_to_use = [torch.nn.ReLU()]
            else:
                raise ValueError(
                    f'Unsupported embedding activation "{embedding_activation}"'
                )
            self.concept_context_generators.append(
                torch.nn.Sequential(*([
                    torch.nn.Linear(
                        list(
                            self.pre_concept_model.modules()
                        )[-1].out_features,
                        # Two as each concept will have a positive and a
                        # negative embedding portion which are later mixed
                        context_gen_out_size,
                    ),
                ] + act_to_use))
            )
            if self.shared_prob_gen and (
                len(self.concept_prob_generators) == 0
            ):
                # Then we will use one and only one probability generator which
                # will be shared among all concepts. This will force concept
                # embedding vectors to be pushed into the same latent space
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
            elif not self.shared_prob_gen:
                self.concept_prob_generators.append(torch.nn.Linear(
                    2 * emb_size,
                    1,
                ))
        if getattr(self, '_construct_c2y_model', True):
            if c2y_model is None:
                # Else we construct it here directly
                units = [
                    n_concepts * emb_size
                ] + (c2y_layers or []) + [n_tasks]
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != len(units) - 1:
                        layers.append(torch.nn.LeakyReLU())
                self.c2y_model = torch.nn.Sequential(*layers)
            else:
                self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights, ignore_index=1000)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.use_concept_groups = use_concept_groups
        self._init_meters()

    def _init_meters(self):
        keys = [
            "diversity",
            "concept_label_dot",
        ]
        self.train_meters = {k: AverageMeter() for k in keys}
        self.val_meters = {k: AverageMeter() for k in keys}

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        **kwargs
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            # Then time to mix!
            bottleneck = (
                pos_embeddings * torch.unsqueeze(prob, dim=-1) +
                neg_embeddings * (1 - torch.unsqueeze(prob, dim=-1))
            )
            return prob, intervention_idxs, bottleneck
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        output = prob * (1 - intervention_idxs) + intervention_idxs * c_true
        # Then time to mix!
        bottleneck = self._construct_c2y_input(
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            probs=output,
            **kwargs,
        )
        return output, intervention_idxs, bottleneck

    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        return self.c2y_model(torch.flatten(bottleneck, start_dim=1, end_dim=-1))

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        **task_loss_kwargs,
    ):
        bottleneck = (
            pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                neg_embeddings * (
                    1 - torch.unsqueeze(probs, dim=-1)
                )
            )
        )
        bottleneck = bottleneck.view(
            (-1, self.n_concepts * self.emb_size)
        )
        return bottleneck

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        pos_embeddings = contexts[:, :, :self.emb_size]
        neg_embeddings = contexts[:, :, self.emb_size:]
        return c_sem, pos_embeddings, neg_embeddings, {}

    def _new_tail_results(
        self,
        x=None,
        c=None,
        y=None,
        c_sem=None,
        bottleneck=None,
        y_pred=None,
    ):
        return []

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )

        c_sem, pos_embs, neg_embs, out_kwargs = self._generate_concept_embeddings(
            x=x,
            latent=latent,
            training=train,
        )

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=pos_embs,
                neg_embeddings=neg_embs,
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )

        else:
            c_int = c
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
                device=x.device,
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        probs, intervention_idxs, bottleneck = self._after_interventions(
            c_sem,
            pos_embeddings=pos_embs,
            neg_embeddings=neg_embs,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
            **out_kwargs
        )
        self._intervention_idxs = intervention_idxs

        y_pred = self._predict_labels(bottleneck=bottleneck)
        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            if "latent" in out_kwargs:
                latent = (latent or tuple([])) + out_kwargs['latent']
            tail_results.append(latent)
        if output_embeddings and (not pos_embs is None) and (
            not neg_embs is None
        ):
            tail_results.append(pos_embs)
            tail_results.append(neg_embs)

        tail_results += self._new_tail_results(
            x=x,
            c=c,
            y=y,
            c_sem=c_sem,
            bottleneck=bottleneck,
            y_pred=y_pred,
        )
        return tuple([c_sem, bottleneck, y_pred] + tail_results)

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
        **kwargs,
    ):
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(batch)
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            **kwargs,
        )
        c_sem, bottleneck, y_logits = outputs[0], outputs[1], outputs[2]

        # ---- extra logging metrics (CEM) ----
        B = c_sem.shape[0]
        _, total_diversity = batch_diversity(c_sem)

        # <c_i, W[y_i, :]> averaged over batch.
        # For CEM, c2y_model typically takes flattened embeddings of size K*emb_size.
        # We derive a per-class per-concept weight by reshaping the (n_tasks, K*emb)
        # matrix into (n_tasks, K, emb) and averaging over emb.
        concept_label_dot = 0.0
        W_concept = None
        linear = None
        if isinstance(self.c2y_model, torch.nn.Sequential):
            # Only use this when the sequential is a single Linear layer.
            if len(self.c2y_model) == 1 and isinstance(self.c2y_model[0], torch.nn.Linear):
                linear = self.c2y_model[0]
        elif isinstance(self.c2y_model, torch.nn.Linear):
            linear = self.c2y_model

        if (linear is not None) and hasattr(linear, "weight"):
            W_flat = linear.weight  # (n_tasks, in_features)
            if W_flat.dim() == 2:
                if W_flat.shape[1] == self.n_concepts:
                    W_concept = W_flat
                elif W_flat.shape[1] == self.n_concepts * self.emb_size:
                    W_concept = W_flat.view(self.n_tasks, self.n_concepts, self.emb_size).mean(dim=-1)

        if (W_concept is not None) and (y is not None):
            y_idx = y.to(dtype=torch.long).reshape(-1)
            valid = (y_idx != 1000)
            if valid.any():
                c_norm = F.normalize(c_sem, dim=1)
                W_y = F.normalize(W_concept[y_idx[valid]], dim=1)
                concept_label_dot = torch.einsum("bk,bk->b", c_norm[valid], W_y).mean().item()

        meters = self.train_meters if train else self.val_meters
        if (meters is not None) and ("diversity" in meters):
            meters["diversity"].update(float(total_diversity.item()), B)
        if (meters is not None) and ("concept_label_dot" in meters):
            meters["concept_label_dot"].update(float(concept_label_dot), int(B))

        # ---- losses ----
        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y.long(),
            )
            task_loss_scalar = task_loss.detach()
        else:
            task_loss = torch.tensor(0.0, device=y_logits.device)
            task_loss_scalar = task_loss.item()

        if self.concept_loss_weight != 0:
            concept_loss = self.loss_concept(c_sem, c)
            concept_loss_scalar = concept_loss.detach()
        else:
            concept_loss = torch.tensor(0.0, device=y_logits.device)
            concept_loss_scalar = concept_loss.item()

        loss = (
            self.concept_loss_weight * concept_loss +
            task_loss +
            self._extra_losses(
                x=x,
                y=y,
                c=c,
                y_pred=y_logits,
                c_sem=c_sem,
                c_pred=bottleneck,
                competencies=competencies,
                prev_interventions=prev_interventions,
            )
        )

        # ---- metrics ----
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
            "diversity": float(total_diversity.item()),
            "concept_label_dot": float(concept_label_dot),
        }
        return loss, result

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = (("auc" in name))
            else:
                prog_bar = (("c_auc" in name) or ("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar)
        result = {"val_" + key: val for key, val in result.items()}
        return result



################################################################################
## Fixed Embedding Version
################################################################################


class FixedEmbConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

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
        use_concept_groups=False,

        context_gen_out_size=None,

        top_k_accuracy=None,

        # New parameters
        fixed_embeddings=True,
        initial_concept_embeddings=None,
        fixed_embeddings_always=True,
    ):
        """
        Same as a CEM but it has a set of learnable global embeddings
        to use for each concept. Useful if you want the interventions
        to be done using a global set of embeddings.
        """

        super(FixedEmbConceptEmbeddingModel, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=shared_prob_gen,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
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
            active_intervention_values=None, # KEY!!!!
            inactive_intervention_values=None, # KEY!!!!
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            use_concept_groups=use_concept_groups,
            context_gen_out_size=context_gen_out_size,
            top_k_accuracy=top_k_accuracy,
        )

        # Let's generate the global embeddings we will use
        if (
            (initial_concept_embeddings is None) and
            (active_intervention_values is not None) and
            (inactive_intervention_values is not None)
        ):
            active_intervention_values = torch.tensor(
                active_intervention_values
            )
            inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )

            initial_concept_embeddings = torch.concat(
                [
                    active_intervention_values.unsqueeze(1),
                    inactive_intervention_values.unsqueeze(1),
                ],
                dim=1,
            )
        self.fixed_embeddings_always = fixed_embeddings_always
        self._set_embeddings = True
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
        self.concept_embeddings = torch.nn.Parameter(
            initial_concept_embeddings,
            requires_grad=(not fixed_embeddings),
        )

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if not self._set_embeddings:
            # Then run the standard CEM pathway
            return ConceptEmbeddingModel._generate_concept_embeddings(
                self=self,
                x=x,
                latent=latent,
                training=training,
            )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, axis=-1)
            contexts = torch.cat(contexts, axis=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent
        if self.fixed_embeddings_always:
            pos_embeddings = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
                x.shape[0],
                -1,
                -1,
            )
            neg_embeddings = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
                x.shape[0],
                -1,
                -1,
            )
        else:
            # Else we only use fixed embeddings for interventions
            pos_embeddings = contexts[:, :, :self.emb_size]
            neg_embeddings = contexts[:, :, self.emb_size:]
        return c_sem, pos_embeddings, neg_embeddings, {}


    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        **kwargs
    ):
        if self.fixed_embeddings_always:
            # Then simply use the CEM pathway
            return ConceptEmbeddingModel._after_interventions(
                self=self,
                prob=prob,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                intervention_idxs=intervention_idxs,
                c_true=c_true,
                train=train,
                competencies=competencies,
                **kwargs
            )
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            # Then time to mix!
            bottleneck = (
                pos_embeddings * torch.unsqueeze(prob, dim=-1) +
                neg_embeddings * (1 - torch.unsqueeze(prob, dim=-1))
            )
            return prob, intervention_idxs, bottleneck
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        output = prob * (1 - intervention_idxs) + intervention_idxs * c_true
        # Use the fixed embeddings for the intervened concepts
        global_pos_embs = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
            intervention_idxs.shape[0],
            -1,
            -1,
        )
        global_neg_embs = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
            intervention_idxs.shape[0],
            -1,
            -1,
        )
        pos_embeddings = pos_embeddings * (1 - intervention_idxs.unsqueeze(-1)) + (
            intervention_idxs.unsqueeze(-1) * global_pos_embs
        )
        neg_embeddings = neg_embeddings * (1 - intervention_idxs.unsqueeze(-1)) + (
            intervention_idxs.unsqueeze(-1) * global_neg_embs
        )
        # Then time to mix!
        bottleneck = self._construct_c2y_input(
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            probs=output,
            **kwargs,
        )
        return output, intervention_idxs, bottleneck