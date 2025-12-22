import torch
import torch.nn.functional as F


class BaseTM:
    def __init__(self, classes, num_clauses, S, T, dtype=torch.float32, device='cuda'):
        self.classes = classes
        self.device = device
        self.dtype = dtype
        self.num_clauses = num_clauses
        self.S = S
        self.T = T
        self.state_min = 0
        self.state_max = 255
        self.include_threshold = (self.state_max + 1) // 2 
        self.clause_size = None

    
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


    # @torch.compile(fullgraph=True,
    #                options={'triton.cudagraphs': True, 
    #                         'shape_padding': True, 
    #                         'max_autotune': True, 
    #                         'epilogue_fusion': True})
    def update(self, x, y_batch, target):
        literals = self.included_literals[y_batch]
        outputs, patch_returns = self.get_clause_output(literals, x)
        class_sum = self.sum_class_votes(outputs)
        ta_states_shift = self.update_clauses(class_sum, target, y_batch, outputs, patch_returns)
        return ta_states_shift
    

    def get_clause_output(self, literals, x, predict=False):
        pass


    def sum_class_votes(self, outputs):
        pass


    def give_literal_feedback(self, mask_shape, device):
        pass



class ConvTM(BaseTM):
    def __init__(self, classes, ks, channels, num_clauses, S, T, dtype=torch.float32, device='cuda'):
        super().__init__(classes, num_clauses, S, T, dtype, device)

        self.ks = ks
        self.channels = channels
        grid_pos_size = 19  # TODO: Need to change to automatically get grid pos size
        self.clause_size = (ks**2)*channels+(grid_pos_size)*4

        # x[:, 0, :, :] -> positive, x[:, 1, :, :] -> negative
        self.ta_states = torch.full(size=(classes, 2, num_clauses//2, self.clause_size),
                                             fill_value=self.include_threshold+1, dtype=dtype, device=device).share_memory_()
        self.included_literals = torch.ones((classes, 2, num_clauses//2, self.clause_size), dtype=dtype, device=device).share_memory_()


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
        type2_mask = (target_vec[:, None] == polarity_idx[None, :]).view(B, 2, 1, 1)
        type1_mask = ~type2_mask

        # Type II feedback
        type2_update = clause_feedback & outputs & ~patch_returns & (batch_states < include_threshold)

        # Type I feedback
        one_literals_mask = clause_feedback & outputs & patch_returns
        zero_literals_mask = clause_feedback & literal_feedback_mask & outputs & ~patch_returns
        zero_outputs_mask = clause_feedback & literal_feedback_mask & ~outputs
        type1_update = one_literals_mask.to(self.dtype) - zero_literals_mask.to(self.dtype) - zero_outputs_mask.to(self.dtype)

        states_add += type2_mask.to(self.dtype) * type2_update.to(self.dtype)
        states_add += type1_mask.to(self.dtype) * type1_update

        new_states = self.ta_states.index_add(0, y_batch, states_add)
        clamped_states = new_states.clamp(self.state_min, self.state_max)

        thresholds = self.include_threshold
        new_literals = (clamped_states >= thresholds).float()
        return clamped_states, new_literals


class ConvCoTM(ConvTM):
    def __init__(self, classes, ks, channels, num_clauses, S, T, dtype, device):
        super().__init__(classes, ks, channels, num_clauses, S, T, dtype, device)

        self.ta_states = torch.full(size=(num_clauses, self.clause_size),
                                    fill_value=self.include_threshold+1, dtype=dtype, device=device).share_memory_()
        self.included_literals = torch.full(size=(num_clauses, self.clause_size), 
                                     fill_value=self.include_threshold+1, dtype=dtype, device=device).share_memory_()
        weight_rands = torch.rand((self.classes, num_clauses), dtype=dtype, device=device)
        self.ta_weights = torch.where(weight_rands < 0.5, -1, 1).to(dtype).share_memory_()


    @torch.compile(fullgraph=True,
                   options={'triton.cudagraphs': True, 
                            'shape_padding': True, 
                            'max_autotune': True, 
                            'epilogue_fusion': True})
    def predict(self, x):
        outputs, _ = self.get_clause_output(x, predict=True)
        class_sum = self.sum_class_votes(outputs)
        return class_sum
    

    # @torch.compile(fullgraph=True,
    #             options={'triton.cudagraphs': True, 
    #                     'shape_padding': True, 
    #                     'max_autotune': True, 
    #                     'epilogue_fusion': True})
    def update(self, x, y_batch):
        # print(self.ta_weights)
        outputs, patch_returns = self.get_clause_output(x)
        # print(outputs)
        class_sum = self.sum_class_votes(outputs)
        # print(class_sum)
        ta_states_shift = self.update_clauses(class_sum, y_batch, outputs, patch_returns)
        return ta_states_shift
    

    def get_clause_output(self, x, predict=False):
        B, P, L = x.shape
        C, _ = self.included_literals.shape

        literals = self.included_literals[None, ...]
        # Number of required included literals per clause per polarity
        required = literals.sum(dim=2)[None, ...]

        # TODO: optimize by storing transposing, shapes, etc in the class definition

        # Number of active patch pixels that correspond to included literals
        outputs = (x @ literals.transpose(1, 2))

        # Clauses activate if the required included literals equal corresponding active patch pixels
        outputs = outputs == required

        # Set empty clauses to true if we are training
        zeroes = required == 0
        outputs = torch.where(zeroes, not predict, outputs)

        # For each true clause, return a random patch index from the patches that the clause activated on
        rand_vals = torch.rand_like(outputs, dtype=self.dtype)
        rand_vals = rand_vals * outputs + outputs - 1
        # TODO: implement below
        ## rand_vals = torch.where(outputs == 1, rand_vals, -1)
        patch_idx = rand_vals.argmax(dim=1)

        # Empty clauses get a random patch sampled from all patches
        rand_all = torch.randint(0, P, patch_idx.shape, dtype=torch.int16, device=self.device)
        patch_idx = torch.where(zeroes, rand_all, patch_idx)

        # print('patch_idx', patch_idx.shape)
        # print('patch_idx.permute(1, 2, 0)', patch_idx.permute(1, 2, 0).shape)
        # print('patch_idx.permute(1, 2, 0).expand(-1, -1, L)', patch_idx.permute(1, 2, 0).expand(-1, -1, L).shape)
        # print('x', x.shape)
        # print('outputs', outputs.shape)

        # Gather patches
        patch_returns = torch.gather(x, 1, patch_idx.permute(1, 2, 0).expand(-1, -1, L))
        # TODO: try below:
        # patch_returns = x[:, patch_idx[0,0], :]
        outputs = outputs.any(dim=1).to(self.dtype)
        # print('patch_returns', patch_returns.shape)
        # print('outputs', outputs.shape)
        # exit()

        return outputs, patch_returns


    def sum_class_votes(self, outputs):
        # print(outputs.shape)
        # print(self.ta_weights.shape)
        # print(outputs @ self.ta_weights.T)
        # exit()
        # class_sum = (outputs[:, 0].sum(1) - outputs[:, 1].sum(1)).clamp(-self.T, self.T)
        class_sum = (outputs @ self.ta_weights.T).clamp(-self.T, self.T)

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


    def update_clauses(self, class_sum, y_batch, outputs, patch_returns):

        B, C = class_sum.shape
        class_sum = class_sum.reshape(-1)

        batch_states = self.ta_states.expand(C * B, -1, -1)
        states_add = torch.zeros_like(batch_states, dtype=self.dtype, device=self.device)
        # _, L = batch_states.shape

        target_mask = F.one_hot(y_batch, num_classes=C).reshape(-1)

        # Mask for per class sum clause feedback
        clause_mask = (self.T + (1 - 2 * target_mask) * class_sum) / (2 * self.T)
        # print(clause_mask)
        # print((1 - 2 * target_mask))
        # print(self.T + (1 - 2 * target_mask) * class_sum)
        # exit()

        # Get clause update masks for all classes
        # TODO: Find way to repeat without the allocation!
        outputs = outputs.repeat(C, 1)
        patch_returns = patch_returns.repeat(C, 1, 1)
        # Get clause feedback
        # print(outputs.shape, clause_mask.shape)
        # exit()
        clause_feedback_mask = torch.rand_like(outputs, dtype=self.dtype, device=self.device) < clause_mask[:, None]
        # print(clause_feedback_mask.shape, batch_states.shape)
        literal_feedback_mask = torch.rand_like(batch_states) < (1.0 / self.S)
        # literal_feedback_mask = self.give_literal_feedback(batch_states.shape, device)
        # print(literal_feedback_mask.shape)
        # exit()
        
        clause_feedback = clause_feedback_mask.unsqueeze(-1)
        outputs = outputs.unsqueeze(-1).to(torch.bool)
        patch_returns = patch_returns.to(torch.bool)
        
        # polarity_idx = torch.arange(2, device=self.device)
        # type2_mask = (target_vec[:, None] == polarity_idx[None, :]).view(B, 2, 1, 1)
        # type1_mask = ~type2_mask
        # print(weight_batch_shape)
        # exit()
        # TODO: expand this once above, happened twice in this function!
        polarity_mask = self.ta_weights.repeat(B, 1) >= 0
        # target_vec = torch.full(polarity_mask.shape, target, device=self.device, dtype=torch.long)
        # The polarity mask for every clause for every class for every batch
        # TODO: Change
        type1_mask = target_mask[..., None] == polarity_mask
        type2_mask = ~type1_mask
        # print(clause_feedback.shape, outputs.shape, patch_returns.shape, batch_states.shape)

        # Type II feedback
        type2_update = clause_feedback & outputs & ~patch_returns & (batch_states < self.include_threshold)

        # Type I feedback
        one_literals_mask = clause_feedback & outputs & patch_returns
        zero_literals_mask = clause_feedback & literal_feedback_mask & outputs & ~patch_returns
        zero_outputs_mask = clause_feedback & literal_feedback_mask & ~outputs
        type1_update = one_literals_mask.to(self.dtype) - zero_literals_mask.to(self.dtype) - zero_outputs_mask.to(self.dtype)
        # print(type1_update.shape, type1_mask.shape, type2_update.shape, type2_mask.shape)
        # exit()

        states_add += type2_mask[..., None].to(self.dtype) * type2_update.to(self.dtype)
        states_add += type1_mask[..., None].to(self.dtype) * type1_update
        # print(states_add.shape)
        # exit()
        # TODO: fix
        # exit()

        # print(type1_update.shape, batch_states.shape[-1])
        # weight_update_magnitude = (type1_update.to(torch.bool) ^ type2_update).any(-1)

                                                                      # TODO: self.L
        # weight_update_scaled = torch.round(weight_update_magnitude / batch_states.shape[-1])
        # print('mag', weight_update_magnitude)
        # print('scaled', weight_update_scaled)
        # exit()
        # print(both_update_mask.shape)
        # print(self.ta_weights.repeat(B, 1).shape)
        # expanded_update_weights = self.ta_weights.repeat(B, 1)
        # print(both_update_mask)
        # print('expanded_update_weights', expanded_update_weights)
        # print('weight_update_magnitude', weight_update_magnitude)
        # exit()
        signed_weights = torch.sign(self.ta_weights)

        # print(signed_weights.shape, type1_update.sum(-1).shape, type2_update.any(-1).shape)
        # exit()
        # print(type1_update.shape, type1_update.to(torch.float32).sum())
        # print(type1_update)
        # print(type1_update.shape)
        
        # exit()
        # TODO: Make sure the reshape here perserves order
        # type1_weight_mask = type1_update.to(torch.float32).sum(-1).reshape(B, C, -1).sum(0)
        # type2_weight_mask = type2_update.to(torch.float32).sum(-1).reshape(B, C, -1).sum(0)
        # random_idxs = torch.randint(0, type1_update.size(-1), (B * C, type1_update.size(-2),), device=self.device)
        # print(random_idxs.shape)
        # print(type1_update[random_idxs[:, None, :]].shape)
        
        random_idxs = torch.randint(0, type1_update.size(-1), (B * C, type1_update.size(-2),), device=self.device)
        type1_weight_mask = type1_update.gather(2, random_idxs.unsqueeze(2)).squeeze(2).reshape(B, C, -1).sum(0)
        random_idxs = torch.randint(0, type1_update.size(-1), (B * C, type1_update.size(-2),), device=self.device)
        type2_weight_mask = type2_update.gather(2, random_idxs.unsqueeze(2)).squeeze(2).reshape(B, C, -1).sum(0)
        # print(type1_weight_mask)
        # exit()
        # print(type2_weight_mask.shape)
        # exit()
        weight_update_1 = torch.sign(signed_weights * type1_weight_mask)
        # print('1', weight_update)
        weight_update_2 = torch.sign(signed_weights * type2_weight_mask)
        weight_update = (weight_update_1 - weight_update_2).to(self.dtype)
        # print('2', weight_update)
        # weight_update_scaled = torch.sign(weight_update)
        zeros = (weight_update == 0)
        # print(weight_update)
        weight_update[zeros] = (torch.rand_like(weight_update[zeros]) < 0.5).to(self.dtype) * 2 - 1
        new_weights = (self.ta_weights + weight_update).clamp(-16, 16)
        # print(weight_update)
        # exit()
        # print('scaled', weight_update)
        # print(weight_update.shape)
        # exit()
        # new_weights = (expanded_update_weights + weight_update).sum(0).squeeze().clamp(-16, 16)
        # new_weights = weight_update.clamp(-16, 16)
        # print(new_weights)
        # print('weight_update', weight_update.shape)
        # print('new_weights', new_weights)
        # exit()
        # print(new_weights.shape)
        # exit()
        # weight_mask = torch.where(type1_mask, 1., -1.).reshape(B, C, -1).to(self.dtype).sum(0).squeeze()
        # weight_add = self.ta_weights + weight_mask

        # new_states = self.ta_states.index_add(0, y_batch, states_add)
        # new_states = (self.ta_states + states_add.sum(0))
        random_index = torch.randint(0, states_add.shape[0], (1,)).item()
        new_states = (self.ta_states + states_add[random_index, ...])
        clamped_states = new_states.clamp(self.state_min, self.state_max)
        new_literals = (clamped_states >= self.include_threshold).float()
        return new_weights, clamped_states, new_literals
    