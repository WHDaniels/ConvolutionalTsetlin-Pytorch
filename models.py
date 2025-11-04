import torch


class ConvTM:
    def __init__(self, classes, ks, channels, num_clauses, S, T, dtype=torch.float32, device='cuda'):
        self.device = device
        self.dtype = dtype
        self.ks = ks
        self.channels = channels
        self.num_clauses = num_clauses
        self.S = S
        self.T = T
        self.state_min = 0
        self.state_max = 255
        self.include_threshold = (self.state_max + 1) // 2 
        grid_pos_size = 19  # TODO: Need to change to automatically get grid pos size
        self.clause_size = (ks**2)*channels+(grid_pos_size)*4

        # x[:, 0, :, :] -> positive, x[:, 1, :, :] -> negative
        self.ta_states = torch.full(size=(classes, 2, num_clauses//2, self.clause_size),
                                             fill_value=self.include_threshold+1, dtype=dtype, device=device).share_memory_()
        self.included_literals = torch.ones((classes, 2, num_clauses//2, self.clause_size), dtype=dtype, device=device).share_memory_()
        

    @torch.compile(fullgraph=True,
                   options={'triton.cudagraphs': True, 
                            'shape_padding': True, 
                            'max_autotune': True, 
                            'epilogue_fusion': True})
    def update(self, x, y_batch, target):
        literals = self.included_literals[y_batch]
        outputs, patch_returns = self.get_clause_output(literals, x)
        class_sum = self.sum_class_votes(outputs)
        ta_states_shift = self.update_clauses(class_sum, target, y_batch, outputs, patch_returns)
        return ta_states_shift
    

    @torch.compile(fullgraph=True,
                   options={'triton.cudagraphs': True, 
                            'shape_padding': True, 
                            'max_autotune': True, 
                            'epilogue_fusion': True})
    def predict(self, x, y_batch):
        literals = self.included_literals[y_batch]
        outputs, _ = self.get_clause_output(literals, x, predict=True)
        class_sum = self.sum_class_votes(outputs)
        return class_sum


    def get_clause_output(self, literals, x, predict=False):
        # TODO: Maybe set order to polarity-first ([B, S, ...] -> [S, B, ...]) for convenience/speed
        B, P, L = x.shape
        _, S, C, _ = literals.shape
        literals = literals.transpose(0, 1).reshape(B * S, C, L)
        x = x.repeat(2, 1, 1)

        # Number of required included literals per clause per polarity
        required = literals.sum(dim=2)

        # Number of active patch pixels that correspond to included literals
        outputs = (x @ literals.transpose(1, 2))

        # Clauses activate if the required included literals equal corresponding active patch pixels
        outputs = outputs == required[:, None, :]

        # Set empty clauses to true if we are training
        zeroes = required[:, None, :] == 0
        outputs = torch.where(zeroes, not predict, outputs)

        # For each true clause, return a random patch index from the patches that the clause activated on
        rand_vals = torch.rand_like(outputs, dtype=literals.dtype)
        rand_vals = rand_vals * outputs + outputs - 1
        patch_idx = rand_vals.argmax(dim=1)

        # Empty clauses get a random patch sampled from all patches
        rand_all = torch.randint(0, P, patch_idx.shape, device=x.device)
        empty_clauses = required == 0
        patch_idx = torch.where(empty_clauses, rand_all, patch_idx)

        # Gather patches
        patch_returns = torch.gather(x, 1, patch_idx.unsqueeze(-1).expand(-1, -1, L))
        patch_returns = patch_returns.reshape(S, B, C, L).transpose(0, 1)

        outputs = outputs.reshape(S, B, P, C).transpose(0, 1).any(dim=2)
        return outputs, patch_returns


    def sum_class_votes(self, outputs):
        class_sum = (outputs[:, 0].sum(1) - outputs[:, 1].sum(1)).clamp(-self.T, self.T)
        return class_sum

    def give_literal_feedback(self, mask_shape, device):
        B, S, C, L = mask_shape
        n = self.clause_size

        # Probability per clause
        p = 1.0 / self.S

        # Number of active literals per clause
        mean, std = n * p, (n * p * (1 - p))
        active = torch.normal(mean, std, size=(C,), device=device).round().int().clamp(0, n)

        # Random permutation mask
        rand = torch.rand(B, S, C, L, device=device)
        topk = rand.argsort(dim=3, descending=True)
        mask = torch.arange(L, device=device).view(1, 1, 1, L) < active.view(1, 1, C, 1)
        feedback_mask = torch.gather(mask.expand(B, S, C, L), 3, topk)
        return feedback_mask

    def update_clauses(self, class_sum, target, y_batch, outputs, patch_returns):
        device = outputs.device

        # Shapes
        batch_states = self.ta_states[y_batch]
        states_add = torch.zeros_like(batch_states, dtype=self.dtype, device=device)
        B, _, C, L = batch_states.shape

        # Clause feedback (Type I / II control)
        clause_mask = (self.T + (1 - 2 * target) * class_sum) / (2 * self.T)
        clause_feedback_mask = torch.rand_like(outputs, dtype=self.dtype, device=device) < clause_mask[:, None, None]
        literal_feedback_mask = torch.rand_like(batch_states) < (1.0 / self.S)
        # literal_feedback_mask = self.give_literal_feedback(batch_states.shape, device)

        clause_feedback = clause_feedback_mask.unsqueeze(-1)
        outputs = outputs.unsqueeze(-1).to(torch.bool)
        patch_returns = patch_returns.to(torch.bool)

        include_threshold = self.include_threshold

        target_vec = torch.full((B,), target, device=y_batch.device, dtype=torch.long)

        polarity_idx = torch.arange(2, device=device)
        # type2 (Type II) applies when polarity == target
        type2_mask = (target_vec[:, None] == polarity_idx[None, :]).view(B, 2, 1, 1)
        type1_mask = ~type2_mask

        # --- Type II feedback: Reinforce exclusion for matching polarity ---
        type2_update = clause_feedback & outputs & ~patch_returns & (batch_states < include_threshold)

        # --- Type I feedback: Reinforce inclusion for opposite polarity ---
        one_literals_mask = clause_feedback & outputs & patch_returns
        zero_literals_mask = clause_feedback & literal_feedback_mask & outputs & ~patch_returns
        zero_outputs_mask = clause_feedback & literal_feedback_mask & ~outputs
        type1_update = one_literals_mask.to(self.dtype) - zero_literals_mask.to(self.dtype) - zero_outputs_mask.to(self.dtype)

        # --- Combine them vectorized ---
        states_add += type2_mask.to(self.dtype) * type2_update.to(self.dtype)
        states_add += type1_mask.to(self.dtype) * type1_update

        # --- Aggregate and clamp ---
        new_states = self.ta_states.index_add(0, y_batch, states_add)
        clamped_states = new_states.clamp(self.state_min, self.state_max)

        thresholds = self.include_threshold
        new_literals = (clamped_states >= thresholds).float()
        return clamped_states, new_literals