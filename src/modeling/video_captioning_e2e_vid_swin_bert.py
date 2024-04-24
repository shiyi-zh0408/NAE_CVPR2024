import torch
from typing import Any, Optional, Tuple, Union
from fairscale.nn.misc import checkpoint_wrapper
import random
from src.layers.bert.modeling_bert import BertEmbeddings
from src.utils.number_to_word import number_to_word
import pickle as pkl
from transformers import CLIPModel, AutoTokenizer, CLIPTokenizer
import math
import torch.nn.functional as F
from src.utils.logger import LOGGER as logger


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# create a class to encode the text
class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size, learnable_length, ouput_size, score_emb_num, score_anno_file, train_split):
        super(CLIPTextEncoder, self).__init__()
        # Load configs
        self.score_dict = pkl.load(open(score_anno_file, 'rb'))
        self.train_split = pkl.load(open(train_split, 'rb'))
        self.score_emb_num = score_emb_num
        self.learnable_length = learnable_length
        
        # Set text_encoder and learnable prompt embeddings.
        self.learnable_embeddings = torch.nn.Parameter(torch.randn(1, learnable_length, hidden_size))
        # self.learnable_embeddings = torch.nn.Parameter(torch.randn(1, learnable_length * score_emb_num, hidden_size))
        # [看不懂]这个地方存疑：这里的tokenizer和embedding和bert的tokenizer和embedding应该是不一样的，这样可能会有问题，先试试吧
        self.clip_text_model = CLIPModel.from_pretrained("/videocap/models/clip")
        self.text_encoder = self.clip_text_model.text_model.encoder
        self.embeddings = self.clip_text_model.text_model.embeddings
        self.tokenizer = CLIPTokenizer.from_pretrained("/videocap/models/clip")

        del self.clip_text_model

        # Get score groups and their ranges.
        self.score_groups_word, self.score_groups_float = self.get_score_groups()
    
    def get_score_groups(self):
        # Return 
        #   1. a list of str, each str represents a score group with the same frequency.
        #   2. a list of tuple, each tuple represents the range of a score group.
        # e.g. 
        #   1. ['zero zero potin zero to twenty nine point five', ...]
        #   2. [(0, 29.5), ...]
        # the length of the list equal to 'score_emb_num'
        self.score_list = []
        for k, v in self.score_dict.items():
            if k in self.train_split:
                self.score_list.append(v['final_score'])
        self.score_list.sort()
        self.score_list = [f'{v:.1f}' for v in self.score_list]
        select_score_list = [self.score_list[int(len(self.score_list)/self.score_emb_num * i)] for i in range(self.score_emb_num)]
        select_score_list.append('105')
        score_groups = [f'{number_to_word(v1)} to {number_to_word(v2)}' for v1, v2 in zip(select_score_list[:-1], select_score_list[1:])]
        select_score_list = [(float(v1), float(v2)) for v1, v2 in zip(select_score_list[:-1], select_score_list[1:])]
        score_groups[0] = 'zero zero point zero' + score_groups[0][15:]
        return score_groups, select_score_list
    
    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask [100, 1, 29, 29]
        return mask
    
    def forward(self, x):
        input_ids = self.tokenizer(self.score_groups_word, padding=True, return_tensors="pt")['input_ids'].cuda()
        input_ids = input_ids[:,1:-1] # [100, 11] -> [100, 9]
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=None) # [100, 9, 512]
        learnable_embeddings = self.learnable_embeddings.expand(hidden_states.shape[0], -1, -1) # [1, 20, 512] -> [100, 20, 512]
        # learnable_embeddings = self.learnable_embeddings.reshape(self.score_emb_num, self.learnable_length, -1)
        # TODO: 这里开头结尾的token需要改一下！
        hidden_states = torch.cat((learnable_embeddings, hidden_states), dim=1) # [100, 29, 512]

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        bsz, seq_len = input_shape # 100, 9
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len + self.learnable_length, hidden_states.dtype).to(hidden_states.device)
        # expand attention_mask
        # if attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.text_encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=False,
        )

        return encoder_outputs
        # return hidden_states

# create a cross attention module
class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, output_attentions):
        super(CrossAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.output_attentions = output_attentions
        q_size = hidden_size
        k_size = hidden_size

        self.query = torch.nn.Linear(q_size, self.all_head_size)
        self.key = torch.nn.Linear(k_size, self.all_head_size)
        self.value = torch.nn.Linear(k_size, self.all_head_size)

        self.layernorm = torch.nn.LayerNorm(hidden_size)
        self.feedforward = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # (B, L, hidden_size) -> (B, num_heads, L, hidden_size/num_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, attention_mask=None, head_mask=None):
        v = k # # q [4, 784, 512] k [4, 100, 512]
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer) # [4, 8, 784, 64]
        key_layer = self.transpose_for_scores(mixed_key_layer) # [4, 8, 100, 64]
        value_layer = self.transpose_for_scores(mixed_value_layer) # [4, 8, 100, 64]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [4, 8, 784, 100]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores) # [4, 8, 784, 100]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # [4, 8, 784, 64]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # [4, 784, 512]

        # Add & Norm
        context_layer = context_layer + q
        context_layer = self.layernorm(context_layer)
        
        # Feed Forward
        context_layer = self.feedforward(context_layer)

        # Add & Norm
        context_layer = context_layer + q
        context_layer = self.layernorm(context_layer)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


# create a multi-layers cross attention module
class CrossAttentionModel(torch.nn.Module):
    def __init__(self, num_layers, hidden_size, num_attention_heads, attention_probs_dropout_prob=0, output_attentions=True):
        super(CrossAttentionModel, self).__init__()
        model_list = []
        for i in range(num_layers):
            output_attentions = True if i == num_layers - 1 else False
            model_list.append(CrossAttentionLayer(hidden_size, num_attention_heads, attention_probs_dropout_prob, output_attentions))
        self.model = torch.nn.ModuleList(model_list)

    def forward(self, q, k):
        outputs = (q,) # q [4, 784, 512] k [4, 100, 512]
        for _layer in self.model:
            outputs = _layer(outputs[0], k)
        return outputs


    

class VideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder, tokenizer):
        super(VideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin

        total_params = sum(p.numel() for p in self.swin.parameters())
        logger.info(f'Swin total parameters: {total_params}')

        self.tokenizer = tokenizer

        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

        # [Modified by Shiyi Zhang]
        # CLIPtext encoder
        self.hidden_size = 512
        self.score_emb_num = args.score_emb_num
        # hidden_size和ouput_size先都定为512，以前是self.hidden_size
        self.clip_text_model = CLIPTextEncoder(hidden_size=512, 
                                               learnable_length=5, 
                                               ouput_size=512, 
                                               score_emb_num=self.score_emb_num,
                                               score_anno_file='/videocap/datasets/MTL_AQA_5_en/final_annotations_dict.pkl', 
                                               train_split='/videocap/datasets/MTL_AQA_5_en/train_split_0.pkl')
        total_params = sum(p.numel() for p in self.clip_text_model.parameters())
        logger.info(f'Clip text model total parameters: {total_params}')

        # Cross Attention
        self.cross_att_1 = CrossAttentionModel(num_layers=8, 
                                             hidden_size=self.hidden_size, 
                                             num_attention_heads=8, 
                                             attention_probs_dropout_prob=0, 
                                             output_attentions=True)
        total_params = sum(p.numel() for p in self.cross_att_1.parameters())
        logger.info(f'Cross-attn-1 total parameters: {total_params}')

        # Cross Attention
        self.cross_att_2 = CrossAttentionModel(num_layers=8, 
                                             hidden_size=self.hidden_size, 
                                             num_attention_heads=8, 
                                             attention_probs_dropout_prob=0, 
                                             output_attentions=True)
        total_params = sum(p.numel() for p in self.cross_att_2.parameters())
        logger.info(f'Cross-attn-2 total parameters: {total_params}')

        # Regressor for AQA
        self.regressor_aqa_1 = torch.nn.Linear(self.hidden_size, 128)
        # # self.regressor_aqa_2 = torch.nn.Linear(128, 64)
        self.regressor_aqa_2 = torch.nn.Linear(128, 32)
        # # self.regressor_aqa_3 = torch.nn.Linear(64, 32)
        self.regressor_aqa_3 = torch.nn.Linear(32, 1)
        # # self.regressor_aqa_4 = torch.nn.Linear(32, 1)

        self.regressor_aqa_ce_1 = torch.nn.Linear(self.hidden_size, 256)
        # # self.regressor_aqa_ce_2 = torch.nn.Linear(256, 128)
        self.regressor_aqa_ce_2 = torch.nn.Linear(256, 100)
        # # self.regressor_aqa_ce_3 = torch.nn.Linear(128, 100)

        self.weight_for_prompt = torch.nn.Parameter(torch.ones(self.hidden_size) * 1e-3)
        self.weight_for_vidfeats = torch.nn.Parameter(torch.ones(self.hidden_size) * 1e-3)

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        # Prepare swin inputs and get image features through swin --> (B x 784 x 512)
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        images = images.permute(0, 2, 1, 3, 4) # (B x S x C x H x W) --> (B x C x S x H x W)
        # through the video swin transformer
        vid_feats = self.swin(images)
        if self.use_grid_feat==True:  
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1) # (B x C x S x H x W) --> (B x S x H x W x C) [(B x 16 x 7 x 7 x 1024)] 
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size) # (B x S x H x W x C) --> (B x 784 x 1024)
        vid_feats = self.fc(vid_feats) # (B x 784 x 1024) --> (B x 784 x 512)

        # 改为直接先过regressor计算得分
        scores_pred, loss_aqa_mse = self.get_aqa_loss_mse(kwargs['score'], vid_feats, kwargs['difficulty'])

        # 其中score_emb_num表示prompt的数量，29表示每个prompt的token数(5learn+9类别)，512表示编码器的隐藏层维度
        prompt_emb = self.clip_text_model(0)[0] # (score_emb_num x 14 x 512) prompt embedding from text encoder
        # 使用.mean(1)方法对第二个维度进行平均，以得到每个prompt的平均embedding。这样处理后，prompt embedding的形状变为了(score_emb_num x 512)
        prompt_emb = prompt_emb.mean(1).unsqueeze(0).repeat(B, 1, 1) # (B x score_emb_num x 512)
        
        # Cross Attention-1. K is the video features, Q is the prompt features from CLIPtext encoder. [Modified by Sule]
        # refine the prompt embedding
        prompt_emb_cross, attention_probs = self.cross_att_1(prompt_emb, vid_feats)
        prompt_emb = prompt_emb + self.weight_for_prompt *  prompt_emb_cross
        
        # Cross Attention-2. Q is the video features, K is the prompt features from CLIPtext encoder. [Modified by Shiyi Zhang]
        # refine the visual feats
        vid_feats_cross, attention_probs = self.cross_att_2(vid_feats, prompt_emb) # attention_probs: (B x num_heads x 784 x score_emb_num)
        
        # change the place calculate ce loss
        pred_cls_id, loss_aqa_ce = self.get_aqa_loss_ce(kwargs['score'], vid_feats_cross, self.clip_text_model.score_groups_float)
        
        vid_feats = vid_feats + self.weight_for_vidfeats * vid_feats_cross # (B x 784 x 512) --> (B x 784 x 512)

        # # Calculate the AQA cross-entropy loss [Modified by Shiyi Zhang]
        # loss_aqa_ce = self.get_aqa_loss_ce(kwargs['score'], vid_feats, self.clip_text_model.score_groups_float)
        
        # Prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats
        
        scores_pred = scores_pred.detach().squeeze().tolist()
        if isinstance(scores_pred, float):
            scores_pred = [scores_pred]
        elif isinstance(scores_pred, list):
            scores_pred = scores_pred
        else:
            scores_pred = []
        
        pred_cls_id = pred_cls_id.detach().squeeze().tolist()
        if isinstance(pred_cls_id, float):
            pred_cls_id = [pred_cls_id]
        elif isinstance(pred_cls_id, list):
            pred_cls_id = pred_cls_id
        else:
            pred_cls_id = []

        input_ids = kwargs['input_ids']

        for index in range(B):
            score = scores_pred[index]
            score_str = str(round(score, 2))
            score_str = number_to_word(score_str)
            tokens = self.tokenizer.tokenize(score_str)
            tokens.append('points')
            tokens.append('.')
            assert len(tokens) <= 9

            if len(tokens) < 9:
                padding_len = 9 - len(tokens)
                tokens += [self.tokenizer.pad_token] * padding_len
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            ids = torch.tensor(ids, dtype=torch.long).to(input_ids.device)

            input_list = input_ids[index].tolist()
            index_dot1 = input_list.index(1012, -60)
            index_dot2 = input_list.index(1012, index_dot1 + 1)
            index_dot3 = input_list.index(1012, index_dot2 + 1)
            index_dot4 = input_list.index(1012, index_dot3 + 1)

            input_ids[index][index_dot3+8: index_dot4+1] = ids

        kwargs['input_ids'] = input_ids
        
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        # learn soft attention mask. the input kwargs['attention_mask'] is a fixed hard attention mask. 'learn_att' is the learnable part.
        # the output attention mask is (B x 924 x 924), where the first 140 are the hard attention mask, and the last 784 are the learnable part.
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        # Forward VL transformer [bert]. the output is a tuple of (loss, logits, hidden_states, attentions)
        # loss is the loss of MLM task.
        # logits is the prediction of the masked tokens in MLM task.
        outputs = self.trans_encoder(*args, **kwargs)

        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  # video_attention is (B x 784 x 784), the learnable part.
            outputs = outputs + (loss_sparsity, )   


        return outputs + (loss_aqa_mse, loss_aqa_ce, scores_pred, pred_cls_id, self.weight_for_prompt.mean().item(), self.weight_for_vidfeats.mean().item())
    
    def get_aqa_loss_mse(self, scores_gt, vid_feats, difficulty):
        pred_vid_feats = vid_feats.mean(dim=1) # (B x 784 x 512)  ->  (B x 512)
        B = pred_vid_feats.shape[0]

        scores_pred = F.relu(self.regressor_aqa_1(pred_vid_feats))
        scores_pred = F.relu(self.regressor_aqa_2(scores_pred))
        scores_pred = self.regressor_aqa_3(scores_pred)

        # 乘以难度系数
        difficulty = difficulty.reshape(B, 1)
        scores_pred = scores_pred * difficulty

        loss_aqa_mse = F.mse_loss(scores_pred, scores_gt.unsqueeze(1))
        
        return scores_pred, loss_aqa_mse

    def get_aqa_loss_ce(self, scores_gt, vid_feats, score_groups_float):
        pred_vid_feats = vid_feats.mean(dim=1) # (B x 784 x 512)  ->  (B x 512)
        B = pred_vid_feats.shape[0]

        # calculate ce loss
        cls_gt = torch.zeros(B).cuda()
        for i in range(B):
            for j, (v1, v2) in enumerate(score_groups_float):
                if v1 <= scores_gt[i].item() < v2:
                    cls_gt[i] = j
                    break
        
        class_pred = F.relu(self.regressor_aqa_ce_1(pred_vid_feats)) # (B x 512) -> (B x 256)
        class_pred = self.regressor_aqa_ce_2(class_pred)
        loss_aqa_ce = F.cross_entropy(class_pred, cls_gt.long())

        # compute predicted class IDs
        _, pred_cls_id = torch.max(class_pred, dim=1)
        
        return pred_cls_id, loss_aqa_ce
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length/pretrained_num_tokens
        
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 


    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        scale_factor = int(self.max_img_seq_length/pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None,None,:,:].double())[0,0,:,:].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length*self.max_img_seq_length,1)


    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

        # for _, p in self.clip_text_model.named_parameters():
        #     p.requires_grad = False
        
        # for name, p in self.named_parameters():
        #     if 'trans_encoder' not in name:
        #         p.requires_grad = False