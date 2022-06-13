import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from tqdm import tqdm
from model_vgg19 import *

from dataloader import *



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # METEOR分数所需wordnet
    #     nltk.download('wordnet')
    #     nltk.download('omw-1.4')

    # Load word map (word2ix)
    with open(args.word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    checkpoint = 'autodl-tmp/models/checkpoint_finetune_epoch_3.pth'

    # Load model
    checkpoint = torch.load(checkpoint)
    encoder = EncoderCNN().to(device)

    decoder = DecoderLSTM(decoder_dim=args.decoder_dim, att_dim=args.att_dim, vocab_size=len(word_map),
                          embed_dim=args.embed_dim, dropout=args.dropout).to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    # Normalization transform
    transform = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])
    for beam_size in range(1, 6):
        bleu4, meteor = evaluate(beam_size, encoder, decoder, transform, word_map, rev_word_map)
        print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu4))
        print("\nMETEOR score @ beam size of %d is %.4f." % (beam_size, meteor))


def evaluate(beam_size, encoder, decoder, transform, word_map, rev_word_map):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score, METEOR
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.data_folder, args.data_name, 'TEST', transform=transform),
        batch_size=1, shuffle=False, num_workers=4)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    meteor = 0.
    vocab_size = len(word_map)
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        # 步长为5 一次性计算5个reference，1个hypo的METEOR分数，共5000个[reference, hypotheses]组合，累加，最后求算数平均即除以5000
        # BLEU4则是一次性计算整个语料库的corpus bleu-4
        if i % 5 != 0:
            continue
        k = beam_size

        image = image.to(device)  # (1, 3, 224, 224)

        encoder_out = encoder(image)                                    # [1, 14, 14, encoder_dim]

        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)              # [1, num_pixels, encoder_dim]
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)    # [k, num_pixels, encoder_dim] 保留最大的k个

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)     # [k, 1] 加入start
        seqs = k_prev_words                                                         # [k, 1]

        top_k_scores = torch.zeros(k, 1).to(device)                                 # [k, 1] 初始化为0 全局累计最大的k个值

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h, c = decoder.init_h_c(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)     # [s, embed_dim]

            att_out, _ = decoder.attention(encoder_out, h)              # [s, num_pixels]

            beta = decoder.sigmoid(decoder.f_beta(h))
            att_out = beta * att_out

            h, c = decoder.decode_step(torch.cat([embeddings, att_out], dim=1), (h, c))  # [s, decoder_dim]

            scores = decoder.fc(h)                                      # [s, vocab_size]
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores            # [s, vocab_size] 因为可能出现了end s在变动

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)        # [s]
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # [s]

            prev_word_inds = top_k_words / vocab_size  # [s] 得到的tensor 表示上一个词来自第几个beam 因为view(-1)过 seq[prev]即可获得上一个词
            next_word_inds = top_k_words % vocab_size  # [s] 得到的tensor 表示是具体哪个词

            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1) # [s, step+1] 前后相连

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]                                  # 还可以继续生成的句子的beam下标
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))        # 已结束句子的beam下标

            # 如果有已经结束的句子 那就将句子和分数加入complete_seqs 和 complete_seqs_scores 且减少beam_size
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0:
                break
            seqs = seqs[incomplete_inds]                                        # 未完成部分 seqs只需按序保留
            h = h[prev_word_inds[incomplete_inds].long()]                       # 而h需要保留生成者 因为top_k后的顺序和上一个h顺序不同 不对应
            c = c[prev_word_inds[incomplete_inds].long()]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))               # scores用来找到最大得分的句子，作为最终结果
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

        ref = list(
            map(lambda c: [rev_word_map[w] for w in c if
                           w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        hyp = [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        for r in ref:
            meteor += single_meteor_score(r, hyp)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4, meteor / 25000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoder_dim', type=int, default=768, help='hidden size of LSTM')
    parser.add_argument('--att_dim', type=int, default=512, help='dim of att')
    parser.add_argument('--embed_dim', type=int, default=256, help='dim of word embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='drop out')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size')
    parser.add_argument('--data_folder', type=str,
                        default='autodl-tmp/coco/images', help='dir of coco images')
    parser.add_argument('--data_name', type=str,
                        default='coco_5_cap_per_img_5_min_word_freq', help='data name of json and hdf5')
    parser.add_argument('--word_map_file', type=str,
                        default='autodl-tmp/coco/images/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                        help='word map path')
    args = parser.parse_args()
    main(args)