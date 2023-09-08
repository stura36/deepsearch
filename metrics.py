import torch
import torch.nn.init
from torch.autograd import Variable
from sentence_transformers import util
import lightning.pytorch as pl


class ContrastiveLoss(pl.LightningModule):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.sim = util.cos_sim
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def calc_r_at_k(viz_emb, txt_emb, debug=False):
    """
    Calculates r@k metric for k = 1,5,10
    Including query - image response - text and viceversa
    """
    #Calculate cosine similarity
    img_txt_sims = util.cos_sim(viz_emb, txt_emb)
    img_txt_sims_t = img_txt_sims.t()
    #Find maximum indices
    top1_img = torch.topk(img_txt_sims, k=1, dim=1)
    top5_img = torch.topk(img_txt_sims, k=5, dim=1)
    top10_img = torch.topk(img_txt_sims, k=10, dim=1)

    top1_txt = torch.topk(img_txt_sims_t, k=1, dim=1)
    top5_txt = torch.topk(img_txt_sims_t, k=5, dim=1)
    top10_txt = torch.topk(img_txt_sims_t, k=10, dim=1)
    #Set counters for correct query responses
    true_1_img = 0
    true_5_img = 0
    true_10_img = 0

    for i in range(img_txt_sims.shape[0]):
        if i in top1_img[1][i]:
            true_1_img += 1
        if i in top5_img[1][i]:
            true_5_img += 1
        if i in top10_img[1][i]:
            true_10_img += 1
    #Set counters for correct query responses
    true_1_txt = 0
    true_5_txt = 0
    true_10_txt = 0

    for i in range(img_txt_sims.shape[1]):
        if i in top1_txt[1][i]:
            true_1_txt += 1
        if i in top5_txt[1][i]:
            true_5_txt += 1
        if i in top10_txt[1][i]:
            true_10_txt += 1
    #Calculate ratio of correct query counts
    r_at_1_img = true_1_img / img_txt_sims.shape[0]
    r_at_5_img = true_5_img / img_txt_sims.shape[0]
    r_at_10_img = true_10_img / img_txt_sims.shape[0]

    r_at_1_txt = true_1_txt / img_txt_sims.shape[0]
    r_at_5_txt = true_5_txt / img_txt_sims.shape[0]
    r_at_10_txt = true_10_txt / img_txt_sims.shape[0]

    return r_at_1_img, r_at_5_img, r_at_10_img, r_at_1_txt, r_at_5_txt, r_at_10_txt
